from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH        = _BASE_DIR / "config" / "universe.csv"
UNIVERSE_GLOBAL_PATH = _BASE_DIR / "config" / "universe_global.csv"

# Main regulated markets only — excludes First North, NGM, Spotlight (small/illiquid)
NORDIC_MAIN_BOARDS: frozenset[str] = frozenset({"OMXS", "OSLO", "OMXH", "OMXC"})
US_EXCHANGES: frozenset[str] = frozenset({"NYSE", "NASDAQ"})


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict, ...]:
    rows: list[dict] = []
    with open(UNIVERSE_PATH, newline="", encoding="utf-8") as f:
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


def get_nordic_tickers(exchanges: frozenset[str] = NORDIC_MAIN_BOARDS) -> list[str]:
    """Enabled Nordic tickers in Yahoo Finance format (.ST/.OL/.HE/.CO)."""
    return [
        r["yahoo_ticker"]
        for r in _load_rows()
        if r["enabled"].strip().lower() == "true"
        and r["exchange"] in exchanges
    ]


def get_us_tickers() -> list[str]:
    """Enabled US tickers (NYSE + NASDAQ) from universe_global.csv."""
    return [
        r["yahoo_ticker"]
        for r in _load_global_rows()
        if r["enabled"].strip().lower() == "true"
        and r["exchange"] in US_EXCHANGES
    ]


def get_sector_from_universe(yahoo_ticker: str) -> str | None:
    """Sector string for a ticker, or None if not found."""
    for rows in (_load_rows(), _load_global_rows()):
        for row in rows:
            if row["yahoo_ticker"] == yahoo_ticker:
                s = row.get("sector", "").strip()
                return s if s else None
    return None


def build_sector_map() -> dict[str, str]:
    """Full yahoo_ticker → sector map for all enabled universe stocks."""
    result = {}
    for rows in (_load_rows(), _load_global_rows()):
        for r in rows:
            if r["enabled"].strip().lower() == "true" and r.get("sector", "").strip():
                result[r["yahoo_ticker"]] = r["sector"].strip()
    return result
