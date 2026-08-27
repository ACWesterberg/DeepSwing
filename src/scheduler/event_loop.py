from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from config.settings import settings
from src.agent.event_decision import get_event_decision
from src.agent.event_risk import validate_event_trade
from src.agent.memory import get_store
from src.analysis.event_model import EventContract, build_candidates
from src.analysis.event_screener import screen_event_candidates
from src.data.kalshi import (
    fetch_markets_by_ticker,
    fetch_weather_markets,
    market_result,
    _parse_market,
)
from src.data.weather_forecast import fetch_forecast_discussion, get_forecast_highs
from src.portfolio.simulator import get_portfolio, persist_portfolio
from src.scheduler.scan_loop import emit

logger = logging.getLogger(__name__)

try:
    from financedata.fx import to_sek as _to_sek_fn
    _HAS_FX = True
except ImportError:
    _HAS_FX = False

# Scans must not overlap — a manual trigger and the scheduled job would otherwise
# size against the same cash twice.
_event_scan_lock = threading.Lock()

_recent: dict = {}


def get_recent_event_decisions() -> dict:
    return _recent


def _usd_sek_rate() -> float:
    if _HAS_FX:
        rate = _to_sek_fn(1.0, "USD")
        if rate:
            return rate
    return settings.usd_sek_fallback


def run_event_scan() -> dict:
    """Run one event cycle, never concurrently with another."""
    if not _event_scan_lock.acquire(blocking=False):
        logger.info("Event scan already in progress — skipping")
        return {"market": "events", "candidates": [], "decisions": [], "busy": True}
    try:
        return _run_event_scan()
    finally:
        _event_scan_lock.release()


def _run_event_scan() -> dict:
    """
    One event cycle: settle and mark what is open, then price the board, screen
    for edges that survive fees, and put survivors to each track's model.

    Settlement runs first so equity is current before anything is sized against it.
    """
    logger.info("=== Event scan started ===")
    decisions_log: list[dict] = []

    settled = _settle_and_mark()
    decisions_log.extend(settled)

    contracts = fetch_weather_markets()
    if not contracts:
        return _finish(decisions_log, candidates=[])

    forecasts = get_forecast_highs(contracts)
    if not forecasts:
        logger.info("No forecasts available — no contracts can be priced")
        return _finish(decisions_log, candidates=[])

    candidates = build_candidates(contracts, forecasts)
    screened = screen_event_candidates(candidates)
    if not screened:
        return _finish(decisions_log, candidates=candidates)

    rate = _usd_sek_rate()
    discussions: dict[str, str] = {}

    for candidate in screened:
        series = candidate.contract.series_ticker
        if series not in discussions:
            discussions[series] = fetch_forecast_discussion(series)

        for track in settings.event_tracks:
            event = _consider(track, candidate, discussions[series], rate)
            if event:
                decisions_log.append(event)

    return _finish(decisions_log, candidates=candidates)


def _consider(track: str, candidate, forecast_discussion: str, rate: float) -> Optional[dict]:
    """Put one screened edge to one track's model, then size and open it."""
    portfolio = get_portfolio(track)

    store = get_store(track)
    heuristics_list = store.retrieve(
        ticker=candidate.contract.series_ticker, regime="event", market="events"
    )

    decision = get_event_decision(
        candidate=candidate,
        track=track,
        forecast_discussion=forecast_discussion,
        heuristics_text=store.to_prompt_text(heuristics_list),
    )

    base = {
        "track": track,
        "ticker": candidate.ticker,
        "regime": "event",
        "fair_prob": round(candidate.fair_prob, 4),
        "market_prob": round(candidate.market_prob, 4),
        "edge": round(candidate.edge, 4),
    }

    if decision is None or decision["action"] != "TRADE":
        return {
            **base,
            "action": decision.get("action", "PASS") if decision else "ERROR",
            "confidence": round(decision.get("confidence", 0.0), 2) if decision else 0.0,
            "reasoning": decision.get("reasoning", "") if decision else "",
        }

    open_positions = [
        {
            "contract_ticker": p.ticker,
            "event_ticker": (p.entry_inputs or {}).get("event_ticker", ""),
            "cost_usd": (p.entry_inputs or {}).get("cost_usd", 0.0),
        }
        for p in portfolio.open_positions
    ]
    equity_usd = portfolio.equity / rate

    risk = validate_event_trade(candidate, equity_usd, open_positions)
    if not risk.approved:
        logger.info("[%s] %s risk rejected: %s", track, candidate.ticker, risk.rejection_reason)
        return {
            **base,
            "action": "BLOCKED",
            "confidence": round(decision["confidence"], 2),
            "reasoning": decision["reasoning"],
            "reason": risk.rejection_reason,
        }

    if settings.event_dry_run:
        logger.info(
            "[%s] DRY RUN would buy %d x %s @ %.3f (fair %.3f, net edge %+.3f, $%.2f)",
            track, risk.contracts, candidate.ticker, candidate.market_prob,
            candidate.fair_prob, risk.net_edge, risk.total_usd,
        )
        return {
            **base,
            "action": "DRY_RUN",
            "confidence": round(decision["confidence"], 2),
            "reasoning": decision["reasoning"],
            "reason": f"{risk.contracts} contracts, ${risk.total_usd:.2f} (event_dry_run)",
        }

    position = portfolio.open_contract(
        ticker=candidate.ticker,
        quantity=risk.contracts,
        price=candidate.market_prob * rate,
        fee=risk.fee_usd * rate,
        payout=rate,  # a YES contract pays $1
        reasoning=decision["reasoning"],
        confidence=decision["confidence"],
        entry_inputs={
            **decision.get("entry_inputs", {}),
            "event_ticker": candidate.contract.event_ticker,
            "series_ticker": candidate.contract.series_ticker,
            "cost_usd": risk.cost_usd,
            "fee_usd": risk.fee_usd,
            "ask": candidate.market_prob,
            "fair_prob": candidate.fair_prob,
            "net_edge": risk.net_edge,
            "forecast_high": candidate.forecast_high,
            "sigma": candidate.sigma,
            # Settle at the entry rate so P&L measures the edge, not USD/SEK drift.
            "usd_sek": rate,
        },
    )
    if position is None:
        return {**base, "action": "BLOCKED", "reason": "insufficient cash",
                "confidence": round(decision["confidence"], 2),
                "reasoning": decision["reasoning"]}

    trade_event = {
        **base,
        "action": "BUY",
        "contracts": risk.contracts,
        "entry_price": position.entry_price,
        "confidence": round(decision["confidence"], 2),
        "reasoning": decision["reasoning"],
        "net_edge": round(risk.net_edge, 4),
    }
    emit({"event": "trade_opened", "data": trade_event})
    return trade_event


def _settle_and_mark() -> list[dict]:
    """
    Settle finalised contracts on Kalshi's own result and mark the rest to the
    current mid. Returns a decisions_log entry per settlement.
    """
    tickers: set[str] = set()
    for track in settings.event_tracks:
        tickers.update(p.ticker for p in get_portfolio(track).open_positions)
    if not tickers:
        return []

    raw_markets = fetch_markets_by_ticker(sorted(tickers))
    if not raw_markets:
        logger.info("No market data for %d open contracts — leaving marks as-is", len(tickers))
        return []

    events: list[dict] = []
    for track in settings.event_tracks:
        portfolio = get_portfolio(track)
        marks: dict[str, float] = {}

        for position in list(portfolio.open_positions):
            raw = raw_markets.get(position.ticker)
            if raw is None:
                continue
            rate = (position.entry_inputs or {}).get("usd_sek") or _usd_sek_rate()

            payout = market_result(raw)
            if payout is not None:
                closed = portfolio.settle_contract(
                    position.trade_id, payout * rate,
                    exit_reason="settled_yes" if payout else "settled_no",
                )
                if closed:
                    events.append({
                        "track": track,
                        "ticker": position.ticker,
                        "action": "SETTLED",
                        "regime": "event",
                        "reasoning": f"Kalshi resolved {'YES' if payout else 'NO'}",
                        "reason": f"P&L {closed.pnl:+.2f} SEK",
                        "fair_prob": (position.entry_inputs or {}).get("fair_prob"),
                        "market_prob": (position.entry_inputs or {}).get("ask"),
                    })
                    emit({"event": "trade_closed", "data": {
                        "track": track,
                        "ticker": closed.ticker,
                        "exit_reason": closed.exit_reason,
                        "pnl": round(closed.pnl, 2),
                        "pnl_pct": round(closed.pnl_pct * 100, 2),
                        "exit_price": closed.exit_price,
                    }})
                continue

            parsed = _parse_market(raw, position.entry_inputs.get("series_ticker", ""))
            if parsed is not None and parsed.mid > 0:
                marks[position.ticker] = parsed.mid * rate

        if marks:
            portfolio.mark_positions(marks)
        persist_portfolio(portfolio)

    return events


def _finish(decisions_log: list[dict], candidates: list) -> dict:
    logger.info(
        "=== Event scan complete | %d priced | %d decisions ===",
        len(candidates), len(decisions_log),
    )
    _recent["events"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "decisions": decisions_log,
    }
    _persist_event_decisions(decisions_log)
    return {
        "market": "events",
        "dry_run": settings.event_dry_run,
        "candidates": [c.to_dict() for c in candidates],
        "decisions": decisions_log,
    }


def _persist_event_decisions(decisions: list[dict]) -> None:
    """Write each decision to the DB for the calibration record. Never breaks a scan."""
    if not decisions:
        return
    from src.db import Decision, get_session

    try:
        session = get_session()
        try:
            for d in decisions:
                session.add(Decision(
                    market="events",
                    track=d.get("track", ""),
                    ticker=d.get("ticker", ""),
                    action=d.get("action", ""),
                    confidence=d.get("confidence"),
                    regime=d.get("regime"),
                    reasoning=d.get("reasoning"),
                    block_reason=d.get("reason"),
                    fair_prob=d.get("fair_prob"),
                    market_prob=d.get("market_prob"),
                    edge=d.get("edge"),
                ))
            session.commit()
        finally:
            session.close()
    except Exception as exc:
        logger.warning("Failed to persist event decisions: %s", exc)
