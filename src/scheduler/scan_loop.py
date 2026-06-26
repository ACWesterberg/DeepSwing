from __future__ import annotations

import logging
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
from src.data.news_fetcher import fetch_news_for_ticker
from src.portfolio.simulator import get_portfolio

logger = logging.getLogger(__name__)

MarketType = Literal["nordic", "us"]

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

    watchlist = settings.nordic_watchlist if market == "nordic" else settings.us_watchlist
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
                decisions_log.append({"track": track, "ticker": candidate.ticker, "action": decision.get("action", "HOLD") if decision else "ERROR"})
                continue

            # Risk validation
            open_pos_info = [
                {"ticker": p.ticker, "sector": p.sector}
                for p in portfolio.open_positions
            ]
            risk = validate_trade(
                action="BUY",
                entry_price=candidate.signals.current_price,
                stop_loss=decision["stop_loss"],
                target=decision["target"],
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
                    "reason": risk.rejection_reason,
                })
                continue

            # Open position
            position = portfolio.open_trade(
                ticker=candidate.ticker,
                market=market,
                quantity=risk.quantity,
                entry_price=candidate.signals.current_price,
                stop_loss=decision["stop_loss"],
                target=decision["target"],
                regime=candidate.regime.regime,
                reasoning=decision["reasoning"],
                confidence=decision["confidence"],
                technical_snapshot=tech_snapshot,
                sector=sector,
            )

            if position:
                trade_event = {
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BUY",
                    "entry_price": position.entry_price,
                    "stop_loss": decision["stop_loss"],
                    "target": decision["target"],
                    "confidence": decision["confidence"],
                    "rrr": risk.rrr,
                    "sector": sector,
                }
                decisions_log.append(trade_event)
                _emit({"event": "trade_opened", "data": trade_event})

    # --- Update open positions and trigger ERL for closed trades ---
    for track in settings.tracks:
        portfolio = get_portfolio(track)
        current_prices = _get_current_prices(portfolio.get_open_tickers(), market)
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

    return {
        "market": market,
        "candidates": [c.to_dict() for c in candidates],
        "decisions": decisions_log,
    }


def _get_current_prices(tickers: list[str], market: str) -> dict[str, float]:
    prices: dict[str, float] = {}
    for ticker in tickers:
        price = get_current_price(ticker, market)
        if price is not None:
            prices[ticker] = price
    return prices


def _trigger_erl(track: str, closed_trade) -> None:
    """Run ERL analysis on a just-closed trade (non-blocking best-effort)."""
    try:
        trade_dict = closed_trade.to_dict()
        trade_dict["id"] = closed_trade.trade_id
        trade_dict["stop_hit"] = closed_trade.exit_reason == "stop_loss"
        trade_dict["pnl_pct"] = closed_trade.pnl_pct

        run_erl(
            track=track,
            trade=trade_dict,
            technicals_str=closed_trade.technical_snapshot,
            regime_str=closed_trade.regime,
        )
    except Exception as exc:
        logger.error("ERL trigger error for %s trade %s: %s", track, closed_trade.trade_id, exc)
