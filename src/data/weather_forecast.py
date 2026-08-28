from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import pytz

from config.settings import settings
from src.analysis.event_model import EventContract

logger = logging.getLogger(__name__)

# Station coordinates must match the station each Kalshi series actually settles
# on. Confirmed 2026-08-28 for KXHIGHNY only: its rules_primary reads "the maximum
# temperature recorded at New York City (CLINYC)", and CLINYC is the NWS climate
# report for Central Park — so these coordinates are right. THE OTHER FIVE ARE
# STILL UNCONFIRMED; scripts/check_event_sources.py prints rules_primary per
# series. A wrong station is a silently biased model, not an error.
#
# Note the same rules settle "according to The Weather Company", not the NWS.
# The underlying observation is the same Central Park climate report, but the
# market is likely pricing off TWC's forecast while we forecast off the NWS.
# Where the two providers disagree, that difference shows up as an apparent edge
# that is really just a difference of opinion between forecasters.


@dataclass(frozen=True)
class WeatherStation:
    name: str
    latitude: float
    longitude: float
    timezone: str


_STATIONS: dict[str, WeatherStation] = {
    "KXHIGHNY": WeatherStation("New York (Central Park)", 40.7789, -73.9692, "America/New_York"),
    "KXHIGHCHI": WeatherStation("Chicago (Midway)", 41.7859, -87.7524, "America/Chicago"),
    "KXHIGHMIA": WeatherStation("Miami (Intl)", 25.7932, -80.2906, "America/New_York"),
    "KXHIGHDEN": WeatherStation("Denver (Intl)", 39.8467, -104.6564, "America/Denver"),
    "KXHIGHLAX": WeatherStation("Los Angeles (Intl)", 33.9425, -118.4081, "America/Los_Angeles"),
    "KXHIGHAUS": WeatherStation("Austin (Camp Mabry)", 30.3210, -97.7600, "America/Chicago"),
}

# /points responses are static per coordinate — cache for the process lifetime.
_points_cache: dict[str, tuple[str, str]] = {}          # series -> (forecast_url, office)
_forecast_cache: dict[str, tuple[datetime, dict[date, float]]] = {}
_discussion_cache: dict[str, tuple[datetime, str]] = {}

_cooldown_until: Optional[datetime] = None


def _available() -> bool:
    return not (_cooldown_until and datetime.now(timezone.utc) < _cooldown_until)


def _trip_breaker(exc: Exception) -> None:
    global _cooldown_until
    _cooldown_until = datetime.now(timezone.utc) + timedelta(
        minutes=settings.kalshi_cooldown_minutes
    )
    logger.warning("NWS request failed (%s) — backing off", exc)


def reset_breaker() -> None:
    """Clear the cooldown and caches — for tests and manual retries."""
    global _cooldown_until
    _cooldown_until = None
    _points_cache.clear()
    _forecast_cache.clear()
    _discussion_cache.clear()


def get_station(series_ticker: str) -> Optional[WeatherStation]:
    return _STATIONS.get(series_ticker.upper())


_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def target_date(contract: EventContract) -> Optional[date]:
    """
    The local calendar day a contract covers.

    Taken from the event ticker (KXHIGHDEN-26AUG27 -> 2026-08-27), which names
    the day directly. close_time is only a fallback: a market stays open past
    the day it covers — settlement waits on the official climate report the next
    morning — so KXHIGHDEN-26AUG27 has a close_time on the 28th. Deriving the
    day from it priced yesterday's contract against today's forecast, which
    produced confident, badly wrong fair values (1.000 against a 1c market).
    """
    from_ticker = _date_from_ticker(contract.event_ticker)
    if from_ticker is not None:
        return from_ticker

    station = get_station(contract.series_ticker)
    if station is None:
        return None
    logger.debug("No date in event ticker %s — falling back to close_time",
                 contract.event_ticker)
    utc = pytz.utc.localize(contract.close_time - timedelta(hours=1))
    return utc.astimezone(pytz.timezone(station.timezone)).date()


def _date_from_ticker(event_ticker: str) -> Optional[date]:
    """Parse the -YYMMMDD suffix Kalshi puts on daily event tickers."""
    match = _EVENT_DATE_RE.search((event_ticker or "").upper())
    if match is None:
        return None
    year, month_name, day = match.groups()
    month = _MONTHS.get(month_name)
    if month is None:
        return None
    try:
        return date(2000 + int(year), month, int(day))
    except ValueError:
        return None


def get_forecast_highs(contracts: list[EventContract]) -> dict[str, float]:
    """
    Map each event_ticker to its forecast daily high in degF.

    Events whose series or target day has no forecast are simply absent from the
    result; build_candidates skips those contracts.
    """
    if not _available():
        logger.info("NWS breaker open — no forecasts this scan")
        return {}

    wanted: dict[str, tuple[str, date]] = {}
    for contract in contracts:
        if contract.event_ticker in wanted:
            continue
        if get_station(contract.series_ticker) is None:
            logger.debug("No station configured for series %s", contract.series_ticker)
            continue
        day = target_date(contract)
        if day is None:
            logger.debug("No target date for %s", contract.event_ticker)
            continue
        wanted[contract.event_ticker] = (contract.series_ticker, day)

    needed_series = {series for series, _ in wanted.values()}
    highs_by_series = {s: _forecast_highs_for_series(s) for s in needed_series}

    forecasts: dict[str, float] = {}
    for event_ticker, (series, day) in wanted.items():
        high = highs_by_series.get(series, {}).get(day)
        if high is None:
            logger.debug("No forecast high for %s on %s", series, day)
            continue
        forecasts[event_ticker] = high

    logger.info("NWS: forecasts for %d/%d events", len(forecasts), len(wanted))
    return forecasts


def _forecast_highs_for_series(series_ticker: str) -> dict[date, float]:
    """Daily highs in degF keyed by local date, cached for nws_forecast_ttl_minutes."""
    now = datetime.now(timezone.utc)
    cached = _forecast_cache.get(series_ticker)
    ttl = settings.nws_forecast_ttl_minutes * 60
    if cached and (now - cached[0]).total_seconds() < ttl:
        return cached[1]

    station = get_station(series_ticker)
    if station is None:
        return {}

    resolved = _resolve_point(series_ticker, station)
    if resolved is None:
        return {}
    forecast_url, _ = resolved

    try:
        payload = _get_json(forecast_url, params={"units": "us"})
    except Exception as exc:
        _trip_breaker(exc)
        return {}

    highs = _parse_periods(payload, station)
    _forecast_cache[series_ticker] = (now, highs)
    return highs


def _parse_periods(payload: dict, station: WeatherStation) -> dict[date, float]:
    """Daytime period temperatures are the forecast highs for their local day."""
    tz = pytz.timezone(station.timezone)
    highs: dict[date, float] = {}

    for period in (payload.get("properties") or {}).get("periods") or []:
        if not period.get("isDaytime"):
            continue
        temperature = period.get("temperature")
        start = period.get("startTime")
        if temperature is None or not start:
            continue
        try:
            when = datetime.fromisoformat(start.replace("Z", "+00:00"))
            value = float(temperature)
        except (AttributeError, TypeError, ValueError):
            continue
        if (period.get("temperatureUnit") or "F").upper() == "C":
            value = value * 9.0 / 5.0 + 32.0
        local_day = when.astimezone(tz).date() if when.tzinfo else when.date()
        highs[local_day] = value

    return highs


def _resolve_point(series_ticker: str, station: WeatherStation) -> Optional[tuple[str, str]]:
    """(hourly-agnostic forecast URL, forecast office) for a station."""
    cached = _points_cache.get(series_ticker)
    if cached:
        return cached

    url = f"{settings.nws_api_base.rstrip('/')}/points/{station.latitude},{station.longitude}"
    try:
        payload = _get_json(url)
    except Exception as exc:
        _trip_breaker(exc)
        return None

    properties = payload.get("properties") or {}
    forecast_url = properties.get("forecast")
    office = properties.get("gridId") or ""
    if not forecast_url:
        logger.warning("NWS /points gave no forecast URL for %s", station.name)
        return None

    _points_cache[series_ticker] = (forecast_url, office)
    return forecast_url, office


def fetch_forecast_discussion(series_ticker: str, max_chars: int = 2000) -> str:
    """
    The forecast office's Area Forecast Discussion — plain-language reasoning about
    confidence, fronts and model disagreement.

    This is the one input that makes the LLM veto worth running: it is where a
    forecaster says the high is uncertain in a way a point forecast cannot. Purely
    best-effort; an empty string is fine.
    """
    now = datetime.now(timezone.utc)
    cached = _discussion_cache.get(series_ticker)
    ttl = settings.nws_forecast_ttl_minutes * 60
    if cached and (now - cached[0]).total_seconds() < ttl:
        return cached[1]

    station = get_station(series_ticker)
    if station is None or not _available():
        return ""
    resolved = _resolve_point(series_ticker, station)
    if resolved is None:
        return ""
    _, office = resolved
    if not office:
        return ""

    base = settings.nws_api_base.rstrip("/")
    try:
        listing = _get_json(f"{base}/products/types/AFD/locations/{office}")
        products = listing.get("@graph") or []
        if not products:
            return ""
        product = _get_json(f"{base}/products/{products[0]['id']}")
        text = (product.get("productText") or "").strip()[:max_chars]
    except Exception as exc:
        logger.debug("Forecast discussion unavailable for %s: %s", office, exc)
        return ""

    _discussion_cache[series_ticker] = (now, text)
    return text


def _get_json(url: str, params: Optional[dict] = None) -> dict:
    headers = {
        "User-Agent": settings.nws_user_agent,
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=settings.nws_timeout_seconds) as client:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
