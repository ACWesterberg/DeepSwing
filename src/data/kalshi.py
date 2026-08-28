from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from config.settings import settings
from src.analysis.event_model import EventContract

logger = logging.getLogger(__name__)

# Verified against the live API on 2026-08-28 via scripts/check_event_sources.py:
# host external-api.kalshi.com, prices served as `*_dollars` decimal strings and
# open interest as `open_interest_fp`. Strike types in use are between/greater/
# less. Re-run that script after any Kalshi API change — a renamed price field
# parses as 0.00 and every market is then rejected as unquoted, which looks like
# a quiet market rather than a broken parser.

_CENTS = 100.0
_PAGE_LIMIT = 200

# Shared failure breaker: every series hits the same host, so one outage would
# otherwise cost a full timeout per series on every scan.
_cooldown_until: Optional[datetime] = None


def _available() -> bool:
    return not (_cooldown_until and datetime.now(timezone.utc) < _cooldown_until)


def _trip_breaker(exc: Exception) -> None:
    global _cooldown_until
    _cooldown_until = datetime.now(timezone.utc) + timedelta(
        minutes=settings.kalshi_cooldown_minutes
    )
    logger.warning(
        "Kalshi request failed (%s) — skipping Kalshi for %d min",
        exc, settings.kalshi_cooldown_minutes,
    )


def reset_breaker() -> None:
    """Clear the cooldown — for tests and for a manual retry after an outage."""
    global _cooldown_until
    _cooldown_until = None


def fetch_weather_markets(
    series: Optional[list[str]] = None,
    days_ahead: Optional[int] = None,
    now: Optional[datetime] = None,
) -> list[EventContract]:
    """
    Open weather contracts resolving within days_ahead, across the configured series.

    Returns an empty list rather than raising when Kalshi is unreachable — a scan
    with no markets is a no-op, not a failure.
    """
    if not _available():
        logger.info("Kalshi breaker open — no event markets this scan")
        return []

    series_list = series if series is not None else settings.event_series
    horizon_days = days_ahead if days_ahead is not None else settings.event_days_ahead
    now = now or datetime.utcnow()
    cutoff = now + timedelta(days=horizon_days)

    contracts: list[EventContract] = []
    for series_ticker in series_list:
        for raw in _fetch_series(series_ticker):
            contract = _parse_market(raw, series_ticker)
            if contract is None:
                continue
            if not now <= contract.close_time <= cutoff:
                continue
            contracts.append(contract)

    logger.info(
        "Kalshi: %d open contracts within %d days across %d series",
        len(contracts), horizon_days, len(series_list),
    )
    return contracts


def fetch_markets_by_ticker(tickers: list[str]) -> dict[str, dict]:
    """
    Raw market objects keyed by ticker, for marking and settling open positions.

    Settlement reads Kalshi's own `result` rather than re-deriving the observed
    high from weather data: the exchange's result is what a real position would
    actually have paid, including any rule quirk our model does not know about.
    """
    if not tickers or not _available():
        return {}

    base = settings.kalshi_api_base.rstrip("/")
    found: dict[str, dict] = {}
    try:
        with httpx.Client(timeout=settings.kalshi_timeout_seconds) as client:
            for ticker in tickers:
                response = client.get(f"{base}/markets/{ticker}")
                if response.status_code == 404:
                    logger.warning("Kalshi has no market %s", ticker)
                    continue
                response.raise_for_status()
                market = (response.json() or {}).get("market")
                if market:
                    found[ticker] = market
    except Exception as exc:
        _trip_breaker(exc)
        return found

    return found


def market_result(raw: dict) -> Optional[float]:
    """
    Settlement payout per contract in USD, or None while the market is still live.

    Kalshi reports `result` as "yes"/"no" once a market finalises; anything else
    means it has not settled yet.
    """
    result = (raw.get("result") or "").strip().lower()
    if result == "yes":
        return 1.0
    if result == "no":
        return 0.0
    return None


def _fetch_series(series_ticker: str) -> list[dict]:
    """All open markets for one series, following the cursor to the last page."""
    markets: list[dict] = []
    cursor: Optional[str] = None
    url = f"{settings.kalshi_api_base.rstrip('/')}/markets"

    try:
        with httpx.Client(timeout=settings.kalshi_timeout_seconds) as client:
            while True:
                params = {
                    "series_ticker": series_ticker,
                    "status": "open",
                    "limit": _PAGE_LIMIT,
                }
                if cursor:
                    params["cursor"] = cursor
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()

                page = payload.get("markets") or []
                markets.extend(page)

                cursor = payload.get("cursor") or None
                # Kalshi echoes a cursor on the final page too; stop on a short
                # page so a repeating cursor can't spin forever.
                if not cursor or len(page) < _PAGE_LIMIT:
                    break
    except Exception as exc:
        _trip_breaker(exc)
        return []

    return markets


def _parse_market(raw: dict, series_ticker: str) -> Optional[EventContract]:
    """Convert one raw market to an EventContract, or None if it is unusable."""
    ticker = raw.get("ticker")
    close_raw = raw.get("close_time")
    if not ticker or not close_raw:
        logger.debug("Skipping market with no ticker/close_time: %s", raw.get("ticker"))
        return None

    close_time = _parse_time(close_raw)
    if close_time is None:
        logger.debug("Skipping %s: unparseable close_time %r", ticker, close_raw)
        return None

    floor_strike = _opt_float(raw.get("floor_strike"))
    cap_strike = _opt_float(raw.get("cap_strike"))
    if floor_strike is None and cap_strike is None:
        logger.debug("Skipping %s: no strike bounds", ticker)
        return None

    return EventContract(
        ticker=ticker,
        event_ticker=raw.get("event_ticker") or ticker.rsplit("-", 1)[0],
        series_ticker=series_ticker,
        title=raw.get("title") or raw.get("subtitle") or ticker,
        yes_bid=_price(raw, "yes_bid_dollars", "yes_bid"),
        yes_ask=_price(raw, "yes_ask_dollars", "yes_ask"),
        last_price=_price(raw, "last_price_dollars", "last_price"),
        volume=_count(raw, "volume", "volume_fp"),
        open_interest=_count(raw, "open_interest", "open_interest_fp"),
        close_time=close_time,
        strike_type=(raw.get("strike_type") or "").lower(),
        floor_strike=floor_strike,
        cap_strike=cap_strike,
    )


def _price(raw: dict, *names: str) -> float:
    """
    A quote as a probability (0..1), from whichever field name this API build uses.

    Kalshi serves prices two ways: `*_dollars` as a decimal string already in
    dollars ("0.9900"), and the older bare field as an integer count of cents.
    Dividing a dollars field by 100 silently yields ~0 for every market, so the
    unit is taken from the field name rather than guessed from the value.
    """
    for name in names:
        value = raw.get(name)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not name.endswith("_dollars"):
            number /= _CENTS
        return max(0.0, min(1.0, number))
    return 0.0


def _count(raw: dict, *names: str) -> int:
    """Volume/open interest, which the `_fp` variants serve as decimal strings."""
    for name in names:
        value = raw.get(name)
        if value is None or value == "":
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _opt_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: str) -> Optional[datetime]:
    """RFC3339 to naive UTC — the rest of the codebase compares against utcnow()."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)
