from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Literal, Optional

from config.settings import settings
from src.agent.decision import get_decision
from src.agent.erl import run_erl
from src.agent.memory import get_store
from src.agent.news_analyzer import analyze_news
from src.agent.risk import validate_trade
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.technical import compute_signals
from src.data.insider_fetcher import get_insider_summary
from src.data.macro_data import get_macro_context
from src.data.market_data import fetch_batch_nordic, fetch_batch_us, get_current_price, get_sector, get_vix
from src.data.watchlist import get_omxs30_tickers
from src.data.news_fetcher import fetch_news_for_ticker
from src.portfolio.simulator import get_portfolio

logger = logging.getLogger(__name__)

MarketType = Literal["nordic", "us"]

# Optional financedata integrations — gracefully absent until installed on the Pi
try:
    from financedata.live import get_live_prices as _get_live_prices
    _HAS_LIVE = True
except ImportError:
    _HAS_LIVE = False

try:
    from financedata.fx import to_sek as _to_sek_fn
    _HAS_FX = True
except ImportError:
    _HAS_FX = False


# financedata returns prices in each stock's native currency — currency mapping
# (exchange → currency) is the calling project's responsibility, not financedata's.
_NORDIC_SUFFIX_CURRENCY: dict[str, str] = {
    ".ST": "SEK",
    ".OL": "NOK",
    ".HE": "EUR",
    ".CO": "DKK",
}


def _currency_for_ticker(ticker: str, market: str) -> str:
    """Resolve the native currency a ticker's price is quoted in."""
    if market != "nordic":
        return "USD"
    for suffix, currency in _NORDIC_SUFFIX_CURRENCY.items():
        if ticker.endswith(suffix):
            return currency
    # Legacy .STO suffix or unrecognized Nordic ticker — assume Swedish (SEK)
    return "SEK"


def _to_sek_price(price: float, ticker: str, market: str) -> float:
    """Convert a ticker's native-currency price to SEK."""
    currency = _currency_for_ticker(ticker, market)
    if currency == "SEK" or not _HAS_FX:
        return price
    sek = _to_sek_fn(price, currency)
    if sek is None:
        logger.warning("%s→SEK FX rate unavailable; using raw %s price %.4f", currency, currency, price)
        return price
    return sek

# Most recent scan decisions per market — ephemeral, in-memory, for dashboard display.
_recent_decisions: dict[str, dict] = {}


def get_recent_decisions() -> dict:
    """Return the latest scan decisions keyed by market."""
    return _recent_decisions


def _persist_decisions(market: str, decisions: list[dict]) -> None:
    """Write each decision to the DB for browsable history. Never breaks a scan."""
    if not decisions:
        return
    from src.db import Decision, get_session

    try:
        session = get_session()
        try:
            for d in decisions:
                session.add(Decision(
                    market=market,
                    track=d.get("track", ""),
                    ticker=d.get("ticker", ""),
                    action=d.get("action", ""),
                    confidence=d.get("confidence"),
                    rrr=d.get("rrr"),
                    regime=d.get("regime"),
                    reasoning=d.get("reasoning"),
                    block_reason=d.get("reason"),
                ))
            session.commit()
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Failed to persist decisions for %s: %s", market, exc)


# Optional callback for pushing trade events to the dashboard WebSocket.
# Injected by app.py on startup; called synchronously from the scan loop thread.
_on_trade_event: Optional[Callable[[dict], None]] = None


def set_trade_event_handler(fn: Callable[[dict], None]) -> None:
    global _on_trade_event
    _on_trade_event = fn


def _emit(event: dict) -> None:
    if _on_trade_event is not None:
        try:
            _on_trade_event(event)
        except Exception as exc:
            logger.warning("Trade event callback error: %s", exc)


def run_scan(market: MarketType) -> dict:
    """
    Full scan cycle for a given market:
    1. Fetch OHLCV for watchlist
    2. Compute technicals + regime
    3. Screen candidates
    4. For each candidate × each track: get decision, validate risk, open trade
    5. Update open positions and trigger ERL for any closed trades

    Returns a summary dict for the dashboard.
    """
    logger.info("=== Scan started: %s market ===", market)

    # VIX circuit-breaker: halt new entries under extreme volatility
    vix = get_vix()
    if vix is not None and vix >= settings.vix_halt_threshold:
        logger.warning(
            "VIX=%.1f >= threshold %.1f — halting new entries for %s market",
            vix, settings.vix_halt_threshold, market,
        )
        return {"market": market, "candidates": [], "decisions": [], "vix_halt": True, "vix": vix}

    watchlist = get_omxs30_tickers() if market == "nordic" else settings.us_watchlist
    macro_context = get_macro_context(market)

    # --- Fetch OHLCV ---
    if market == "nordic":
        ohlcv_map = fetch_batch_nordic(watchlist)
    else:
        ohlcv_map = fetch_batch_us(watchlist)

    if not ohlcv_map:
        logger.warning("No OHLCV data returned for %s market", market)
        return {"market": market, "candidates": [], "decisions": []}

    # --- Compute technicals + regime ---
    analysis_map: dict[str, tuple] = {}
    for ticker, df in ohlcv_map.items():
        signals = compute_signals(ticker, df)
        if signals is None:
            continue
        regime = classify_regime(df)
        analysis_map[ticker] = (signals, regime)

    # --- Screen ---
    candidates = screen_candidates(analysis_map, market)
    if not candidates:
        logger.info("No candidates passed screener for %s", market)
        return {"market": market, "candidates": [], "decisions": []}

    # --- Decision + risk + execution per candidate × track ---
    decisions_log = []

    for candidate in candidates:
        # Shared data: news, insider
        articles = fetch_news_for_ticker(candidate.ticker, market)
        insider_summary = get_insider_summary(candidate.ticker, market)
        tech_brief = f"Price {candidate.signals.current_price:.4f}, RSI {candidate.signals.rsi_14:.1f}"

        news_summary = analyze_news(
            ticker=candidate.ticker,
            market=market,
            current_price=candidate.signals.current_price,
            technicals_brief=tech_brief,
            articles=articles,
        )

        full_news = f"{news_summary}\nInsider activity: {insider_summary}"

        tech_snapshot = candidate.signals.to_prompt_str()
        sector = get_sector(candidate.ticker)

        for track in settings.tracks:
            portfolio = get_portfolio(track)

            # Retrieve heuristics
            store = get_store(track)
            heuristics_list = store.retrieve(
                ticker=candidate.ticker,
                regime=candidate.regime.regime,
                market=market,
            )
            heuristics_text = store.to_prompt_text(heuristics_list)

            # Get AI decision
            decision = get_decision(
                candidate=candidate,
                track=track,
                news_summary=full_news,
                macro_context=macro_context,
                heuristics_text=heuristics_text,
            )

            if decision is None or decision["action"] != "BUY":
                logger.debug("[%s] %s → %s", track, candidate.ticker, decision.get("action") if decision else "None")
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": decision.get("action", "HOLD") if decision else "ERROR",
                    "confidence": round(decision.get("confidence", 0.0), 2) if decision else 0.0,
                    "reasoning": decision.get("reasoning", "") if decision else "",
                    "regime": candidate.regime.regime,
                })
                continue

            # Convert prices to SEK from the ticker's native currency
            entry_sek = _to_sek_price(candidate.signals.current_price, candidate.ticker, market)
            stop_sek = _to_sek_price(decision["stop_loss"], candidate.ticker, market)
            target_sek = _to_sek_price(decision["target"], candidate.ticker, market)

            # Risk validation
            open_pos_info = [
                {"ticker": p.ticker, "sector": p.sector}
                for p in portfolio.open_positions
            ]
            risk = validate_trade(
                action="BUY",
                entry_price=entry_sek,
                stop_loss=stop_sek,
                target=target_sek,
                portfolio_equity=portfolio.equity,
                open_positions=open_pos_info,
                signals=candidate.signals,
                is_drawdown_mode=portfolio.is_drawdown_mode,
                candidate_sector=sector,
            )

            if not risk.approved:
                logger.info("[%s] %s risk rejected: %s", track, candidate.ticker, risk.rejection_reason)
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": round(risk.rrr, 2) if risk.rrr else None,
                    "regime": candidate.regime.regime,
                    "reason": risk.rejection_reason,
                })
                continue

            # Open position
            position = portfolio.open_trade(
                ticker=candidate.ticker,
                market=market,
                quantity=risk.quantity,
                entry_price=entry_sek,
                stop_loss=stop_sek,
                target=target_sek,
                regime=candidate.regime.regime,
                reasoning=decision["reasoning"],
                confidence=decision["confidence"],
                technical_snapshot=tech_snapshot,
                sector=sector,
                entry_inputs=decision.get("entry_inputs", {}),
            )

            if position:
                trade_event = {
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BUY",
                    "entry_price": position.entry_price,
                    "stop_loss": stop_sek,
                    "target": target_sek,
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": round(risk.rrr, 2),
                    "regime": candidate.regime.regime,
                    "sector": sector,
                }
                decisions_log.append(trade_event)
                _emit({"event": "trade_opened", "data": trade_event})

    # --- Update open positions and trigger ERL for closed trades ---
    for track in settings.tracks:
        portfolio = get_portfolio(track)
        # Fetch prices per position's own market — mixing markets here causes wrong FX conversion
        positions_by_market: dict[str, list[str]] = {}
        for p in portfolio.open_positions:
            positions_by_market.setdefault(p.market, []).append(p.ticker)
        current_prices: dict[str, float] = {}
        for pos_market, pos_tickers in positions_by_market.items():
            current_prices.update(_get_current_prices(pos_tickers, pos_market))
        closed_trades = portfolio.update_prices(current_prices)

        for closed in closed_trades:
            _emit({
                "event": "trade_closed",
                "data": {
                    "track": track,
                    "ticker": closed.ticker,
                    "exit_reason": closed.exit_reason,
                    "pnl_pct": round(closed.pnl_pct * 100, 2),
                    "pnl": round(closed.pnl, 2),
                    "exit_price": closed.exit_price,
                },
            })
            _trigger_erl(track, closed)

    logger.info("=== Scan complete: %s | %d candidates | %d decisions ===",
                market, len(candidates), len(decisions_log))

    _recent_decisions[market] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decisions": decisions_log,
    }
    _persist_decisions(market, decisions_log)

    return {
        "market": market,
        "candidates": [c.to_dict() for c in candidates],
        "decisions": decisions_log,
    }


def _get_current_prices(tickers: list[str], market: str) -> dict[str, float]:
    if not tickers:
        return {}
    if _HAS_LIVE:
        yf_tickers = [t.replace(".STO", ".ST") if market == "nordic" else t for t in tickers]
        ticker_map = dict(zip(yf_tickers, tickers))
        try:
            raw = _get_live_prices(yf_tickers)
            return {
                ticker_map[yf]: _to_sek_price(price, ticker_map[yf], market)
                for yf, price in raw.items()
                if price is not None and yf in ticker_map
            }
        except Exception as exc:
            logger.warning("get_live_prices failed, falling back: %s", exc)
    prices: dict[str, float] = {}
    for ticker in tickers:
        price = get_current_price(ticker, market)
        if price is not None:
            prices[ticker] = _to_sek_price(price, ticker, market)
    return prices


def _trigger_erl(track: str, closed_trade) -> None:
    """Run ERL analysis on a just-closed trade (non-blocking best-effort)."""
    try:
        trade_dict = closed_trade.to_dict()
        trade_dict["id"] = closed_trade.trade_id
        trade_dict["stop_hit"] = closed_trade.exit_reason == "stop_loss"
        trade_dict["pnl_pct"] = closed_trade.pnl_pct

        # Entry-time news/macro so ERL can attribute outcomes to the environment
        entry_inputs = getattr(closed_trade, "entry_inputs", {}) or {}

        run_erl(
            track=track,
            trade=trade_dict,
            technicals_str=closed_trade.technical_snapshot,
            regime_str=closed_trade.regime,
            news_str=entry_inputs.get("news_summary", ""),
            macro_str=entry_inputs.get("macro_context", ""),
        )
    except Exception as exc:
        logger.error("ERL trigger error for %s trade %s: %s", track, closed_trade.trade_id, exc)
