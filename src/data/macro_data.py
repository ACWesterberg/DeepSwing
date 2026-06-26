from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Cache macro data — it changes slowly
_macro_cache: dict[str, tuple[datetime, str]] = {}
_CACHE_TTL_HOURS = 6


def get_macro_context(market: str) -> str:
    """Return a short macro context string for the given market."""
    cache_key = market
    cached = _macro_cache.get(cache_key)
    if cached:
        ts, text = cached
        if (datetime.utcnow() - ts).total_seconds() < _CACHE_TTL_HOURS * 3600:
            return text

    if market == "us":
        text = _build_us_macro()
    else:
        text = _build_nordic_macro()

    _macro_cache[cache_key] = (datetime.utcnow(), text)
    return text


def _build_us_macro() -> str:
    parts = []

    # FRED: Federal Funds Rate
    ffr = _fetch_fred("FEDFUNDS")
    if ffr:
        parts.append(f"Fed Funds Rate: {ffr:.2f}%")

    # FRED: CPI YoY
    cpi = _fetch_fred("CPIAUCSL")
    if cpi:
        parts.append(f"CPI (latest): {cpi:.1f}")

    # FRED: 10Y Treasury
    t10y = _fetch_fred("DGS10")
    if t10y:
        parts.append(f"10Y Treasury: {t10y:.2f}%")

    # FRED: Unemployment Rate
    unemp = _fetch_fred("UNRATE")
    if unemp:
        parts.append(f"Unemployment: {unemp:.1f}%")

    if not parts:
        return "US macro data unavailable."

    return "US Macro | " + " | ".join(parts)


def _build_nordic_macro() -> str:
    parts = []

    # Riksbank policy rate (public API)
    rate = _fetch_riksbank_rate()
    if rate:
        parts.append(f"Riksbank Rate: {rate:.2f}%")

    # ECB rate as proxy for EUR context
    ecb = _fetch_ecb_rate()
    if ecb:
        parts.append(f"ECB Rate: {ecb:.2f}%")

    if not parts:
        return "Nordic macro data unavailable."

    return "Nordic Macro | " + " | ".join(parts)


def _fetch_fred(series_id: str) -> Optional[float]:
    if not settings.fred_api_key:
        return None

    since = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={settings.fred_api_key}"
        f"&file_type=json&sort_order=desc&limit=1&observation_start={since}"
    )

    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        if obs and obs[0].get("value") not in (".", None):
            return float(obs[0]["value"])
    except Exception as exc:
        logger.debug("FRED error for %s: %s", series_id, exc)
    return None


def _fetch_riksbank_rate() -> Optional[float]:
    """Fetch Riksbank policy rate from public SWEA API."""
    try:
        url = "https://api.riksbank.se/swea/v1/Observations/SECBREPOEFF/2023-01-01"
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        observations = resp.json()
        if observations:
            return float(observations[-1].get("value", 0))
    except Exception as exc:
        logger.debug("Riksbank API error: %s", exc)
    return None


def _fetch_ecb_rate() -> Optional[float]:
    """Fetch ECB deposit facility rate from ECB data warehouse."""
    try:
        url = (
            "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.DFR.LEV"
            "?lastNObservations=1&format=jsondata"
        )
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        series = data.get("dataSets", [{}])[0].get("series", {})
        if series:
            obs = next(iter(series.values())).get("observations", {})
            if obs:
                latest = max(obs.keys(), key=int)
                return float(obs[latest][0])
    except Exception as exc:
        logger.debug("ECB API error: %s", exc)
    return None
