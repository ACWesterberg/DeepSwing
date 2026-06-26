from __future__ import annotations

import logging
from datetime import datetime, time
from typing import Literal

import pytz

logger = logging.getLogger(__name__)

MarketType = Literal["nordic", "us"]

_TZ_CET = pytz.timezone("Europe/Stockholm")
_TZ_ET = pytz.timezone("America/New_York")

# Scan windows in CET
_NORDIC_OPEN = time(8, 45)
_NORDIC_CLOSE = time(17, 45)

_US_OPEN_CET = time(15, 15)
_US_CLOSE_CET = time(22, 15)

# Market is closed on weekends
_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon–Fri


def is_market_open(market: MarketType, dt: datetime | None = None) -> bool:
    """Return True if the given market has an active scan window at dt (default: now)."""
    now_cet = _now_cet(dt)

    if now_cet.weekday() not in _WEEKDAYS:
        return False

    current_time = now_cet.time()

    if market == "nordic":
        return _NORDIC_OPEN <= current_time <= _NORDIC_CLOSE

    if market == "us":
        return _US_OPEN_CET <= current_time <= _US_CLOSE_CET

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
    now_cet = _now_cet()
    if market == "nordic":
        open_time = _NORDIC_OPEN
    else:
        open_time = _US_OPEN_CET

    if now_cet.time() < open_time and now_cet.weekday() in _WEEKDAYS:
        day = now_cet.strftime("%A")
        return f"{day} at {open_time.strftime('%H:%M')} CET"

    # Next weekday
    days_ahead = 1
    while (now_cet.weekday() + days_ahead) % 7 not in _WEEKDAYS:
        days_ahead += 1
    return f"In {days_ahead} day(s) at {open_time.strftime('%H:%M')} CET"


def _now_cet(dt: datetime | None = None) -> datetime:
    if dt is None:
        dt = datetime.utcnow().replace(tzinfo=pytz.utc)
    elif dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(_TZ_CET)
