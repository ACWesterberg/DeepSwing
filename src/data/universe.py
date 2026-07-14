from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH        = _BASE_DIR / "config" / "universe.csv"
UNIVERSE_EU_PATH     = _BASE_DIR / "config" / "universe_eu.csv"
UNIVERSE_GLOBAL_PATH = _BASE_DIR / "config" / "universe_global.csv"

# Main regulated markets only — excludes First North, NGM, Spotlight (small/illiquid)
NORDIC_MAIN_BOARDS: frozenset[str] = frozenset({"OMXS", "OSLO", "OMXH", "OMXC"})
EU_MAIN_BOARDS: frozenset[str] = frozenset({
    "LSE", "XETRA", "EURONEXT", "BME", "SIX", "WSE", "VIE",
})
US_EXCHANGES: frozenset[str] = frozenset({"NYSE", "NASDAQ"})


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict, ...]:
    rows: list[dict] = []
    with open(UNIVERSE_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return tuple(rows)


@lru_cache(maxsize=1)
def _load_eu_rows() -> tuple[dict, ...]:
    rows: list[dict] = []
    if not UNIVERSE_EU_PATH.is_file():
        return tuple()
    with open(UNIVERSE_EU_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return tuple(rows)


@lru_cache(maxsize=1)
def _load_global_rows() -> tuple[dict, ...]:
    rows: list[dict] = []
    with open(UNIVERSE_GLOBAL_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return tuple(rows)


def _enabled_tickers(rows: tuple[dict, ...], exchanges: frozenset[str]) -> list[str]:
    return [
        r["yahoo_ticker"]
        for r in rows
        if r["enabled"].strip().lower() == "true"
        and r["exchange"] in exchanges
    ]


def get_nordic_tickers(exchanges: frozenset[str] = NORDIC_MAIN_BOARDS) -> list[str]:
    """Enabled Nordic tickers in Yahoo Finance format (.ST/.OL/.HE/.CO)."""
    return _enabled_tickers(_load_rows(), exchanges)


def get_eu_tickers(exchanges: frozenset[str] = EU_MAIN_BOARDS) -> list[str]:
    """Enabled continental-EU tickers (.L/.DE/.PA/…) from universe_eu.csv."""
    return _enabled_tickers(_load_eu_rows(), exchanges)


def get_us_tickers() -> list[str]:
    """Enabled US tickers (NYSE + NASDAQ) from universe_global.csv."""
    return _enabled_tickers(_load_global_rows(), US_EXCHANGES)


def get_name_from_universe(yahoo_ticker: str) -> str | None:
    """Company name for a ticker (e.g. 'Volvo B' for VOLV-B.ST), or None."""
    for rows in (_load_rows(), _load_eu_rows(), _load_global_rows()):
        for row in rows:
            if row["yahoo_ticker"] == yahoo_ticker:
                name = row.get("name", "").strip()
                return name if name else None
    return None


def get_sector_from_universe(yahoo_ticker: str) -> str | None:
    """Sector string for a ticker, or None if not found."""
    for rows in (_load_rows(), _load_eu_rows(), _load_global_rows()):
        for row in rows:
            if row["yahoo_ticker"] == yahoo_ticker:
                s = row.get("sector", "").strip()
                return s if s else None
    return None


_SUFFIX_EXCHANGE: dict[str, str] = {
    ".ST": "OMXS",
    ".OL": "OSLO",
    ".HE": "OMXH",
    ".CO": "OMXC",
    ".L": "LSE",
    ".DE": "XETRA",
    ".PA": "EURONEXT",
    ".AS": "EURONEXT",
    ".BR": "EURONEXT",
    ".MC": "BME",
    ".SW": "SIX",
    ".WA": "WSE",
    ".VI": "VIE",
    ".LS": "EURONEXT",
    ".IR": "EURONEXT",
}


def get_exchange_from_universe(yahoo_ticker: str) -> str | None:
    """Exchange code for a ticker (e.g. OMXS, NASDAQ), or None if not in universe."""
    for rows in (_load_rows(), _load_eu_rows(), _load_global_rows()):
        for row in rows:
            if row["yahoo_ticker"] == yahoo_ticker:
                ex = row.get("exchange", "").strip()
                return ex if ex else None
    for suffix, exchange in _SUFFIX_EXCHANGE.items():
        if yahoo_ticker.endswith(suffix):
            return exchange
    return None


def get_exchange_for_ticker(yahoo_ticker: str, market: str = "") -> str:
    """Exchange label for dashboard display; falls back to market or suffix."""
    exchange = get_exchange_from_universe(yahoo_ticker)
    if exchange:
        return exchange
    if market == "us":
        return "US"
    if market == "eu":
        return "EU"
    if market == "nordic":
        return "Nordic"
    return "—"


def build_sector_map() -> dict[str, str]:
    """Full yahoo_ticker → sector map for all enabled universe stocks."""
    result = {}
    for rows in (_load_rows(), _load_eu_rows(), _load_global_rows()):
        for r in rows:
            if r["enabled"].strip().lower() == "true" and r.get("sector", "").strip():
                result[r["yahoo_ticker"]] = r["sector"].strip()
    return result
