from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from config.settings import settings
from src.agent.decision import get_decision, get_exit_decision
from src.agent.erl import run_erl
from src.agent.memory import get_store
from src.agent.news_analyzer import analyze_news
from src.agent.risk import compute_return_correlations, validate_trade
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.triage import triage_candidates
from src.analysis.technical import compute_signals
from src.data.insider_fetcher import get_insider_summary
from src.data.macro_data import get_macro_context
from src.data.market_data import fetch_batch_eu, fetch_batch_nordic, fetch_batch_us, get_current_price, get_days_to_earnings, get_sector, get_vix
from src.data.watchlist import get_eu_watchlist, get_omxs30_tickers, get_us_tickers
from src.data.news_fetcher import fetch_market_headlines, fetch_news_for_ticker, format_market_environment
from src.portfolio.simulator import get_portfolio, persist_portfolio

logger = logging.getLogger(__name__)

from src.scheduler.markets import MarketType

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

_EU_SUFFIX_CURRENCY: dict[str, str] = {
    ".L": "GBP",
    ".DE": "EUR",
    ".PA": "EUR",
    ".AS": "EUR",
    ".BR": "EUR",
    ".MC": "EUR",
    ".SW": "CHF",
    ".WA": "PLN",
    ".VI": "EUR",
    ".LS": "EUR",
    ".IR": "EUR",
}


def _currency_for_ticker(ticker: str, market: str, strict: bool = False) -> Optional[str]:
    """Resolve the native currency a ticker's price is quoted in.

    strict=True returns None instead of guessing when the ticker doesn't match
    its market (a suffixed listing in the US watchlist, an unknown EU suffix) —
    a wrong currency assumption corrupts every SEK figure booked for the trade.
    Non-strict keeps the lenient guess so existing positions can still be priced
    and exited consistently with how they were entered."""
    if market == "us":
        if strict and "." in ticker:
            return None
        return "USD"
    suffix_map = _EU_SUFFIX_CURRENCY if market == "eu" else _NORDIC_SUFFIX_CURRENCY
    for suffix, currency in suffix_map.items():
        if ticker.endswith(suffix):
            return currency
    if market == "eu":
        return None if strict else "EUR"
    if ticker.endswith(".STO"):  # legacy Alpha Vantage suffix — Swedish
        return "SEK"
    return None if strict else "SEK"


def _to_sek_price(price: float, ticker: str, market: str, strict: bool = False) -> Optional[float]:
    """Convert a ticker's native-currency price to SEK. Returns None when the
    conversion is unavailable — booking a raw USD/EUR price against a SEK
    portfolio corrupts sizing and P&L, so callers must skip instead."""
    currency = _currency_for_ticker(ticker, market, strict=strict)
    if currency is None:
        logger.error(
            "Cannot resolve quote currency for %s in %s market — skipping", ticker, market,
        )
        return None
    if currency == "SEK":
        return price
    if not _HAS_FX:
        logger.error(
            "%s is quoted in %s but FX conversion is unavailable (financedata.fx missing) — skipping",
            ticker, currency,
        )
        return None
    sek = _to_sek_fn(price, currency)
    if sek is None:
        logger.warning("%s→SEK FX rate unavailable for %s — skipping price", currency, ticker)
        return None
    return sek

# Most recent scan decisions per market — ephemeral, in-memory, for dashboard display.
_recent_decisions: dict[str, dict] = {}


def get_recent_decisions() -> dict:
    """Return the latest scan decisions keyed by market."""
    return _recent_decisions


def clear_recent_decisions() -> None:
    _recent_decisions.clear()


def _persist_decisions(market: str, decisions: list[dict]) -> None:
    """Write each decision to the DB for browsable history. Never breaks a scan."""
    if not decisions:
        return
    from src.db import Decision, get_session

    try:
        session = get_session()
        try:
            # One entry_inputs blob per (track, ticker) per day is enough for
            # counterfactual training — a ticker gets re-decided every 15 min,
            # and persisting every copy would bloat the DB fast on the Pi.
            day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            stored_today = {
                (row.track, row.ticker)
                for row in session.query(Decision.track, Decision.ticker)
                .filter(Decision.timestamp >= day_start, Decision.entry_inputs.isnot(None))
                .all()
            }
            for d in decisions:
                inputs = d.get("entry_inputs")
                key = (d.get("track", ""), d.get("ticker", ""))
                if inputs and key in stored_today:
                    inputs = None
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
                    price=d.get("price"),
                    atr=d.get("atr"),
                    entry_inputs=inputs,
                ))
                if inputs:
                    stored_today.add(key)
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


# Serialize scans across threads — the scheduler and a manual /api/scan trigger
# must never run concurrently, or two scans could double-open the same ticker.
_scan_lock = threading.Lock()


def run_scan(market: MarketType) -> dict:
    """Run a scan, but never concurrently with another scan (see _scan_lock)."""
    if not _scan_lock.acquire(blocking=False):
        logger.info("Scan already in progress — skipping %s scan", market)
        return {"market": market, "candidates": [], "decisions": [], "busy": True}
    try:
        return _run_scan(market)
    finally:
        _scan_lock.release()


def _run_scan(market: MarketType) -> dict:
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

    # VIX circuit-breaker: halt new entries under extreme volatility — but keep
    # managing open positions (stops/targets/news exits); abandoning holdings in
    # a volatility spike is exactly when stop enforcement matters most.
    vix = get_vix()
    if vix is not None and vix >= settings.vix_halt_threshold:
        logger.warning(
            "VIX=%.1f >= threshold %.1f — halting new entries for %s market (holdings still monitored)",
            vix, settings.vix_halt_threshold, market,
        )
        result = _monitor_holdings(market)
        result["vix_halt"] = True
        result["vix"] = vix
        return result

    # If no track has budget to open a new position *in this market*, the
    # candidate/news/decision pipeline can't produce a trade — skip it and just
    # monitor open holdings. Budget is gated per market (market_allocation) so a
    # long US session can't fill the whole book and starve the Nordic session.
    funded_tracks = [t for t in settings.tracks if get_portfolio(t).can_open_in_market(market)]
    if not funded_tracks:
        logger.info("No track has %s-market budget available — holdings-only monitor", market)
        return _monitor_holdings(market)

    if market == "nordic":
        watchlist = get_omxs30_tickers()
    elif market == "eu":
        watchlist = get_eu_watchlist()
    else:
        watchlist = get_us_tickers()
    logger.info("Watchlist: %d tickers for %s market", len(watchlist), market)
    if not watchlist:
        logger.warning("Empty watchlist for %s market — nothing to scan", market)
        return {"market": market, "candidates": [], "decisions": []}
    macro_market = "nordic" if market in ("nordic", "eu") else "us"
    macro_context = get_macro_context(macro_market)

    # Market-wide news environment (geopolitics, sector themes, risk sentiment) —
    # fetched once per scan and folded into the macro context so it reaches the
    # decision model, ERL, and MIPRO without a signature change.
    try:
        market_env = format_market_environment(
            fetch_market_headlines(market, limit=settings.market_news_max_headlines)
        )
        macro_context = f"{macro_context}\n\n{market_env}"
    except Exception as exc:
        logger.warning("Market-wide news fetch failed: %s", exc)

    # --- Fetch OHLCV ---
    if market == "nordic":
        ohlcv_map = fetch_batch_nordic(watchlist)
    elif market == "eu":
        ohlcv_map = fetch_batch_eu(watchlist)
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

    # --- Earnings-proximity filter: never trade into an earnings gap ---
    candidates = _filter_earnings(candidates)
    if not candidates:
        logger.info("All candidates filtered out by earnings proximity for %s", market)
        return {"market": market, "candidates": [], "decisions": []}

    # --- Cheap shared triage: only the top-K reach news + per-track decisions ---
    candidates = triage_candidates(candidates, market)

    # Live quotes for entry fills. The OHLCV close a candidate was screened on
    # can be hours stale (Alpha Vantage is end-of-day, EU feeds are delayed);
    # booking entries at it while exits fill at live prices realizes the gap as
    # phantom P&L the moment the same-scan stop/target sweep runs.
    entry_quotes_sek = _get_current_prices([c.ticker for c in candidates], market)

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

        # Only tracks with cash to deploy get an entry decision this scan.
        for track in funded_tracks:
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
                entry = {
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": decision.get("action", "PASS") if decision else "ERROR",
                    "confidence": round(decision.get("confidence", 0.0), 2) if decision else 0.0,
                    "reasoning": decision.get("reasoning", "") if decision else "",
                    "regime": candidate.regime.regime,
                }
                # PASS decisions carry their price + DSPy inputs so MIPRO can later
                # label passed-on setups from subsequent price data (counterfactuals).
                if decision and decision["action"] == "PASS":
                    entry["price"] = candidate.signals.current_price
                    entry["atr"] = candidate.signals.atr_14
                    entry["entry_inputs"] = decision.get("entry_inputs")
                decisions_log.append(entry)
                continue

            # Convert prices to SEK from the ticker's native currency — strict,
            # so a ticker whose quote currency can't be resolved is blocked
            # instead of booked at a guessed FX rate.
            signal_sek = _to_sek_price(candidate.signals.current_price, candidate.ticker, market, strict=True)
            stop_sek = _to_sek_price(decision["stop_loss"], candidate.ticker, market, strict=True)
            target_sek = _to_sek_price(decision["target"], candidate.ticker, market, strict=True)
            if signal_sek is None or stop_sek is None or target_sek is None or signal_sek <= 0:
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "regime": candidate.regime.regime,
                    "reason": "FX conversion to SEK unavailable",
                })
                continue

            # Fill entries at the live quote, never the scan-time OHLCV close.
            # If no live quote exists, or it has drifted past the deviation cap,
            # the screened setup no longer describes the market — block.
            entry_sek = entry_quotes_sek.get(candidate.ticker)
            deviation = abs(entry_sek - signal_sek) / signal_sek if entry_sek is not None else None
            if entry_sek is None or deviation > settings.max_entry_price_deviation:
                reason = (
                    "Live entry price unavailable"
                    if entry_sek is None
                    else (
                        f"Live price {entry_sek:.4f} SEK deviates {deviation:.1%} from scan price "
                        f"{signal_sek:.4f} (max {settings.max_entry_price_deviation:.0%}) — signals are stale"
                    )
                )
                logger.info("[%s] %s entry blocked: %s", track, candidate.ticker, reason)
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "regime": candidate.regime.regime,
                    "reason": reason,
                })
                continue

            # Risk validation — correlations vs same-market open positions come
            # from the batch OHLCV already fetched this scan (no extra network)
            open_pos_info = [
                {"ticker": p.ticker, "sector": p.sector}
                for p in portfolio.open_positions
            ]
            correlations = compute_return_correlations(
                ohlcv_map.get(candidate.ticker),
                [p.ticker for p in portfolio.open_positions if p.market == market],
                ohlcv_map,
            )
            # Cap sizing at the market's remaining allocation budget (<= cash), so a
            # single scan can't blow past this market's share and starve the other.
            investable = portfolio.market_budget_remaining(market)
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
                available_cash=investable,
                position_correlations=correlations,
            )

            if not risk.approved:
                logger.info("[%s] %s risk rejected: %s", track, candidate.ticker, risk.rejection_reason)
                # Blocked BUYs (e.g. weak targets, now that they're rejected
                # instead of stretched) carry inputs + price/ATR too, so the
                # counterfactual pipeline still learns from setups never taken.
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": round(risk.rrr, 2) if risk.rrr else None,
                    "regime": candidate.regime.regime,
                    "reason": risk.rejection_reason,
                    "price": candidate.signals.current_price,
                    "atr": candidate.signals.atr_14,
                    "entry_inputs": decision.get("entry_inputs"),
                })
                continue

            # Trailing distance in SEK: ATR is native-currency, so scale it by
            # the FX rate the scan price was converted at (entry_sek is a live
            # quote and no longer maps 1:1 to signals.current_price).
            fx_rate = signal_sek / candidate.signals.current_price if candidate.signals.current_price > 0 else 1.0
            trail_distance = settings.trailing_stop_atr_multiplier * candidate.signals.atr_14 * fx_rate

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
                # heuristic_ids ride along (outside the DSPy signature) so the
                # heuristics used at entry can be re-scored against the outcome.
                entry_inputs={
                    **decision.get("entry_inputs", {}),
                    "heuristic_ids": [h["id"] for h in heuristics_list],
                },
                trail_distance=trail_distance,
                entry_fx_rate=fx_rate,
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
            else:
                # Approved by risk but rejected at execution (e.g. cash consumed
                # by an earlier candidate this scan) — record it, don't lose it.
                decisions_log.append({
                    "track": track,
                    "ticker": candidate.ticker,
                    "action": "BLOCKED",
                    "confidence": round(decision["confidence"], 2),
                    "reasoning": decision["reasoning"],
                    "rrr": round(risk.rrr, 2),
                    "regime": candidate.regime.regime,
                    "reason": "Insufficient cash at execution",
                })

    # Live prices for every held ticker, fetched per the position's own market —
    # mixing markets causes wrong FX conversion. Shared by the news-exit review
    # and the stop/target sweep so both fill at real quotes, not scan-time OHLCV
    # closes (which can be hours stale and would book phantom exit P&L).
    held_by_market: dict[str, list[str]] = {}
    for track in settings.tracks:
        for p in get_portfolio(track).open_positions:
            held_by_market.setdefault(p.market, []).append(p.ticker)
    live_prices: dict[str, float] = {}
    for pos_market, pos_tickers in held_by_market.items():
        live_prices.update(_get_current_prices(sorted(set(pos_tickers)), pos_market))

    # --- News-driven exit review, gated on a large price move ---
    # Holdings are otherwise monitored on price alone (stop/target below). We only
    # spend a news pull + AI exit review on a position that has jumped since its
    # last news check — using the fresh batch signals for tickers still in the
    # watchlist. Set holdings_news_jump_pct=0.0 to review every scan.
    for track in settings.tracks:
        portfolio = get_portfolio(track)
        for position in list(portfolio.open_positions):
            if position.market != market or position.ticker not in analysis_map:
                continue
            current_sek = live_prices.get(position.ticker)
            if current_sek is None:
                continue
            signals, regime = analysis_map[position.ticker]
            event = _maybe_news_exit(
                track, portfolio, position, current_sek, market,
                signals_str=signals.to_prompt_str(),
                regime_str=regime.to_prompt_str(),
                regime_label=regime.regime,
                macro_context=macro_context,
            )
            if event:
                decisions_log.append(event)

    # --- Update open positions and trigger ERL for closed trades ---
    for track in settings.tracks:
        portfolio = get_portfolio(track)
        for closed in portfolio.update_prices(live_prices):
            _emit_close(track, closed)
        # End-of-scan flush — captures mark-to-market / trailing-stop updates on
        # positions that didn't close (opens/closes already persisted inline).
        persist_portfolio(portfolio)

    logger.info("=== Scan complete: %s | %d candidates | %d decisions ===",
                market, len(candidates), len(decisions_log))

    # Dashboard/WebSocket payloads don't need the multi-KB DSPy input blobs —
    # those only exist for the counterfactual trainset and go to the DB.
    display_log = [
        {k: v for k, v in d.items() if k != "entry_inputs"} for d in decisions_log
    ]
    _recent_decisions[market] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decisions": display_log,
    }
    _persist_decisions(market, decisions_log)

    return {
        "market": market,
        "candidates": [c.to_dict() for c in candidates],
        "decisions": display_log,
    }


def _monitor_holdings(market: MarketType) -> dict:
    """
    Lightweight cycle used when no track has cash to open a new position: pull
    prices for open holdings, run the jump-gated news exit per position, then the
    stop-loss/take-profit sweep. No watchlist fetch, no candidate/entry pipeline.
    """
    decisions_log: list[dict] = []
    macro_context = get_macro_context(market)

    for track in settings.tracks:
        portfolio = get_portfolio(track)
        positions = [p for p in portfolio.open_positions if p.market == market]
        if not positions:
            continue

        prices = _get_current_prices([p.ticker for p in positions], market)

        # A large price jump triggers a news pull + AI exit review; otherwise we
        # rely on the mechanical stop/target sweep below. No fresh OHLCV here, so
        # the exit review reuses the entry-time technical snapshot.
        for position in list(positions):
            price = prices.get(position.ticker)
            if price is None:
                continue
            event = _maybe_news_exit(
                track, portfolio, position, price, market,
                signals_str=position.technical_snapshot or "No live technicals (holdings-only monitor).",
                regime_str=position.regime,
                regime_label=position.regime,
                macro_context=macro_context,
            )
            if event:
                decisions_log.append(event)

        for closed in portfolio.update_prices(prices):
            _emit_close(track, closed)
        persist_portfolio(portfolio)

    open_count = sum(
        len([p for p in get_portfolio(t).open_positions if p.market == market])
        for t in settings.tracks
    )
    logger.info("=== Holdings monitor: %s | %d open | %d exits ===",
                market, open_count, len(decisions_log))

    _recent_decisions[market] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decisions": decisions_log,
    }
    _persist_decisions(market, decisions_log)

    return {
        "market": market,
        "mode": "holdings_monitor",
        "candidates": [],
        "decisions": decisions_log,
    }


def _maybe_news_exit(
    track: str,
    portfolio,
    position,
    current_price_sek: float,
    market: MarketType,
    signals_str: str,
    regime_str: str,
    regime_label: str,
    macro_context: str,
) -> Optional[dict]:
    """
    News-driven exit for a holding, gated on a large price move. Only when the
    position has moved >= holdings_news_jump_pct since its last news check do we
    pull news and run the AI exit review; a SELL closes it as 'news_exit'. All
    prices are SEK. Returns a decisions_log event if closed, else None.
    """
    ref = position.last_news_price or position.entry_price
    if ref <= 0:
        return None
    move = (current_price_sek - ref) / ref
    if abs(move) < settings.holdings_news_jump_pct:
        return None

    position.last_news_price = current_price_sek
    position.current_price = current_price_sek
    move_pct = move * 100
    days_held = (datetime.utcnow() - position.entry_time).days
    logger.info(
        "[%s] %s moved %+.1f%% since last news check (%.4f→%.4f) — running exit review",
        track, position.ticker, move_pct, ref, current_price_sek,
    )

    articles = fetch_news_for_ticker(position.ticker, market, force_refresh=True)
    news_summary = analyze_news(
        ticker=position.ticker,
        market=market,
        current_price=current_price_sek,
        technicals_brief=(
            f"Held from {position.entry_price:.4f}, now {current_price_sek:.4f} "
            f"({move_pct:+.1f}% since last check)"
        ),
        articles=articles,
    )

    store = get_store(track)
    heuristics_list = store.retrieve(ticker=position.ticker, regime=regime_label, market=market)
    pos_ctx = (
        f"Entry: {position.entry_price:.4f}, Current: {current_price_sek:.4f}, "
        f"P&L: {position.unrealised_pnl_pct * 100:+.2f}%, Stop: {position.stop_loss:.4f}, "
        f"Target: {position.target:.4f}, Days held: {days_held}, "
        f"Move since last news: {move_pct:+.1f}%"
    )
    exit_dec = get_exit_decision(
        ticker=position.ticker,
        market=market,
        track=track,
        signals_str=signals_str,
        regime_str=regime_str,
        position_context=pos_ctx,
        news_summary=news_summary or "No recent news available.",
        macro_context=macro_context,
        heuristics_text=store.to_prompt_text(heuristics_list),
    )
    if not exit_dec or exit_dec["action"] != "SELL":
        return None

    closed = portfolio.close_trade(
        trade_id=position.trade_id,
        exit_price=current_price_sek,
        exit_reason="news_exit",
        regime=regime_label,
        reasoning=exit_dec["reasoning"],
        confidence=exit_dec["confidence"],
    )
    if not closed:
        return None
    logger.info(
        "[%s] news exit: %s at %.4f (P&L %.2f%%) — %s",
        track, position.ticker, current_price_sek, closed.pnl_pct * 100, exit_dec["reasoning"][:80],
    )
    _emit_close(track, closed)
    return {
        "track": track,
        "ticker": position.ticker,
        "action": "SELL",
        "confidence": round(exit_dec["confidence"], 2),
        "reasoning": exit_dec["reasoning"],
        "regime": regime_label,
    }


def _emit_close(track: str, closed) -> None:
    """Broadcast a trade-closed event and run ERL for a just-closed trade."""
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
    _record_heuristic_outcome(track, closed)
    _trigger_erl(track, closed)


def _record_heuristic_outcome(track: str, closed) -> None:
    """Re-score the heuristics that informed this trade's entry by its outcome."""
    try:
        entry_inputs = getattr(closed, "entry_inputs", {}) or {}
        heuristic_ids = entry_inputs.get("heuristic_ids") or []
        if heuristic_ids:
            get_store(track).record_outcome(heuristic_ids, closed.pnl_pct)
    except Exception as exc:
        logger.warning("Heuristic outcome update failed for %s trade %s: %s",
                       track, closed.trade_id, exc)


def _filter_earnings(candidates: list) -> list:
    """Drop candidates within settings.earnings_buffer_days of their earnings date."""
    buffer = settings.earnings_buffer_days
    if buffer <= 0 or not candidates:
        return candidates
    days_map = get_days_to_earnings([c.ticker for c in candidates])
    kept = []
    for c in candidates:
        days = days_map.get(c.ticker)
        if days is not None and days <= buffer:
            logger.info("Earnings filter: skipping %s (earnings in %d day(s))", c.ticker, days)
            continue
        kept.append(c)
    return kept


def _get_current_prices(tickers: list[str], market: str) -> dict[str, float]:
    if not tickers:
        return {}
    if _HAS_LIVE:
        yf_tickers = [t.replace(".STO", ".ST") if market == "nordic" else t for t in tickers]
        ticker_map = dict(zip(yf_tickers, tickers))
        try:
            raw = _get_live_prices(yf_tickers)
            prices: dict[str, float] = {}
            for yf, price in raw.items():
                if price is None or yf not in ticker_map:
                    continue
                sek = _to_sek_price(price, ticker_map[yf], market)
                if sek is not None:
                    prices[ticker_map[yf]] = sek
            return prices
        except Exception as exc:
            logger.warning("get_live_prices failed, falling back: %s", exc)
    prices = {}
    for ticker in tickers:
        price = get_current_price(ticker, market)
        if price is None:
            continue
        sek = _to_sek_price(price, ticker, market)
        if sek is not None:
            prices[ticker] = sek
    return prices


# In-flight ERL threads — ERL is a heavy extended-thinking call, so it runs off
# the scan thread. wait_for_erl() lets tests and shutdown paths drain them.
_erl_threads: list[threading.Thread] = []


def _current_fx_rate(ticker: str, market: str) -> Optional[float]:
    """Current native→SEK rate for a ticker, None when unknown/SEK."""
    currency = _currency_for_ticker(ticker, market)
    if currency is None or currency == "SEK" or not _HAS_FX:
        return None
    return _to_sek_fn(1.0, currency)


def _erl_kwargs(track: str, closed_trade) -> dict:
    """Build the run_erl(**kwargs) payload from a closed trade."""
    trade_dict = closed_trade.to_dict()
    trade_dict["id"] = closed_trade.trade_id
    trade_dict["stop_hit"] = closed_trade.exit_reason == "stop_loss"
    trade_dict["pnl_pct"] = closed_trade.pnl_pct
    # All booked prices are SEK, so part of the P&L can be pure currency drift.
    # Give ERL the FX move over the holding period so an FX-triggered stop
    # isn't misattributed to the stock's own behaviour.
    entry_fx = getattr(closed_trade, "entry_fx_rate", 0.0) or 0.0
    if entry_fx > 0:
        current_fx = _current_fx_rate(closed_trade.ticker, closed_trade.market)
        if current_fx is not None:
            trade_dict["fx_move_pct"] = (current_fx / entry_fx - 1) * 100
    # Entry-time news/macro so ERL can attribute outcomes to the environment
    entry_inputs = getattr(closed_trade, "entry_inputs", {}) or {}
    return dict(
        track=track,
        trade=trade_dict,
        technicals_str=closed_trade.technical_snapshot,
        regime_str=closed_trade.regime,
        news_str=entry_inputs.get("news_summary", ""),
        macro_str=entry_inputs.get("macro_context", ""),
    )


def _trigger_erl(track: str, closed_trade) -> None:
    """Run ERL analysis on a just-closed trade in a background thread."""
    try:
        kwargs = _erl_kwargs(track, closed_trade)
    except Exception as exc:
        logger.error("ERL trigger error for %s trade %s: %s", track, closed_trade.trade_id, exc)
        return

    trade_id = kwargs["trade"]["id"]

    def _worker() -> None:
        try:
            run_erl(**kwargs)
        except Exception as exc:
            logger.error("ERL error for %s trade %s: %s", track, trade_id, exc)

    thread = threading.Thread(target=_worker, name=f"erl-{track}-{trade_id}", daemon=True)
    _erl_threads.append(thread)
    thread.start()


def wait_for_erl(timeout: float = 60.0) -> None:
    """Block until in-flight ERL threads finish (or timeout)."""
    deadline = time.monotonic() + timeout
    for thread in list(_erl_threads):
        thread.join(max(0.0, deadline - time.monotonic()))
    _erl_threads[:] = [t for t in _erl_threads if t.is_alive()]


def backfill_erl(tracks: Optional[list[str]] = None) -> dict:
    """
    Re-run ERL over closed trades that don't yet have a heuristic — e.g. trades
    that closed while the ERL model call was failing. Runs synchronously (unlike
    the live path's background thread) and returns per-track counts.

    Idempotent: a trade whose trade_id already sourced a heuristic is skipped, so
    tracks that already learned from a trade are never re-analyzed or duplicated.
    """
    target = tracks or list(settings.tracks)
    result: dict[str, dict] = {}
    for track in target:
        portfolio = get_portfolio(track)
        store = get_store(track)
        already = {
            h.get("source_trade_id")
            for h in store.all_as_list()
            if h.get("source_trade_id") is not None
        }
        attempted = 0
        created = 0
        for closed in portfolio.closed_trades:
            if closed.trade_id in already:
                continue
            attempted += 1
            try:
                if run_erl(**_erl_kwargs(track, closed)):
                    created += 1
            except Exception as exc:
                logger.error("ERL backfill error for %s trade %s: %s", track, closed.trade_id, exc)
        result[track] = {"trades_without_heuristic": attempted, "heuristics_created": created}
        logger.info("ERL backfill [%s]: %d without heuristic, %d created", track, attempted, created)
    return result
