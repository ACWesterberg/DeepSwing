from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Literal

import exchange_calendars as xcals
import pytz

logger = logging.getLogger(__name__)

MarketType = Literal["nordic", "us"]

_TZ_CET = pytz.timezone("Europe/Stockholm")
_TZ_ET = pytz.timezone("America/New_York")

# Actual exchange hours — Nordic in CET, US in US Eastern Time. US hours must be
# evaluated in ET: the US and EU switch DST weeks apart, so a fixed CET window
# misses the first trading hour (or overshoots the close) during those weeks.
_NORDIC_EXCHANGE_OPEN = time(9, 0)
_NORDIC_EXCHANGE_CLOSE = time(17, 30)
_US_EXCHANGE_OPEN_ET = time(9, 30)
_US_EXCHANGE_CLOSE_ET = time(16, 0)

# Scan windows include a 15-min buffer before open and after close
_NORDIC_OPEN = time(8, 45)
_NORDIC_CLOSE = time(17, 45)
_US_OPEN_ET = time(9, 15)
_US_CLOSE_ET = time(16, 15)

_WEEKDAYS = {0, 1, 2, 3, 4}

# exchange_calendars codes: XSTO = Stockholm, XNYS = NYSE (covers US holidays)
_CAL_NORDIC = xcals.get_calendar("XSTO")
_CAL_US = xcals.get_calendar("XNYS")


def is_market_open(market: MarketType, dt: datetime | None = None) -> bool:
    """Return True if the given market has an active scan window at dt (default: now)."""
    if market == "nordic":
        now_cet = _now_cet(dt)
        if now_cet.weekday() not in _WEEKDAYS:
            return False
        if not _CAL_NORDIC.is_session(now_cet.date().isoformat()):
            return False
        return _NORDIC_OPEN <= now_cet.time() <= _NORDIC_CLOSE

    if market == "us":
        now_et = _now_in(_TZ_ET, dt)
        if now_et.weekday() not in _WEEKDAYS:
            return False
        if not _CAL_US.is_session(now_et.date().isoformat()):
            return False
        return _US_OPEN_ET <= now_et.time() <= _US_CLOSE_ET

    return False


def is_exchange_open(market: MarketType, dt: datetime | None = None) -> bool:
    """True only during official exchange hours — used for display, not scheduling."""
    if market == "nordic":
        now_cet = _now_cet(dt)
        if now_cet.weekday() not in _WEEKDAYS:
            return False
        if not _CAL_NORDIC.is_session(now_cet.date().isoformat()):
            return False
        return _NORDIC_EXCHANGE_OPEN <= now_cet.time() <= _NORDIC_EXCHANGE_CLOSE
    if market == "us":
        now_et = _now_in(_TZ_ET, dt)
        if now_et.weekday() not in _WEEKDAYS:
            return False
        if not _CAL_US.is_session(now_et.date().isoformat()):
            return False
        return _US_EXCHANGE_OPEN_ET <= now_et.time() <= _US_EXCHANGE_CLOSE_ET
    return False


def active_markets(dt: datetime | None = None) -> list[MarketType]:
    """Return list of currently active markets."""
    markets: list[MarketType] = []
    if is_market_open("nordic", dt):
        markets.append("nordic")
    if is_market_open("us", dt):
        markets.append("us")
    return markets


def next_open_cet(market: MarketType) -> str:
    """Return a human-readable string of the next market open in CET."""
    from datetime import timedelta

    # Walk forward in the market's own timezone; display the open time in CET
    # (for US the CET-equivalent open shifts with the DST offset mismatch).
    tz = _TZ_CET if market == "nordic" else _TZ_ET
    open_time = _NORDIC_OPEN if market == "nordic" else _US_OPEN_ET
    cal = _CAL_NORDIC if market == "nordic" else _CAL_US

    now_local = _now_in(tz)
    check = now_local.date()
    if now_local.time() >= open_time:
        check += timedelta(days=1)

    for _ in range(10):
        if cal.is_session(check.isoformat()):
            open_local = tz.localize(datetime.combine(check, open_time))
            open_cet = open_local.astimezone(_TZ_CET)
            days_ahead = (check - now_local.date()).days
            label = "Today" if days_ahead == 0 else f"In {days_ahead} day(s)"
            return f"{label} at {open_cet.strftime('%H:%M')} CET"
        check += timedelta(days=1)

    return f"Unknown (next session within 10 days not found)"


def _now_in(tz, dt: datetime | None = None) -> datetime:
    if dt is None:
        dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(tz)


def _now_cet(dt: datetime | None = None) -> datetime:
    return _now_in(_TZ_CET, dt)
