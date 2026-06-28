from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = _BASE_DIR / "config" / "universe.csv"

# Main regulated markets only — excludes First North, NGM, Spotlight (small/illiquid)
NORDIC_MAIN_BOARDS: frozenset[str] = frozenset({"OMXS", "OSLO", "OMXH", "OMXC"})


@lru_cache(maxsize=1)
def _load_rows() -> tuple[dict, ...]:
    rows: list[dict] = []
    with open(UNIVERSE_PATH, newline="", encoding="utf-8") as f:
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


def get_sector_from_universe(yahoo_ticker: str) -> str | None:
    """Sector string for a ticker, or None if not found."""
    for row in _load_rows():
        if row["yahoo_ticker"] == yahoo_ticker:
            s = row.get("sector", "").strip()
            return s if s else None
    return None


def build_sector_map() -> dict[str, str]:
    """Full yahoo_ticker → sector map for all enabled universe stocks."""
    return {
        r["yahoo_ticker"]: r["sector"].strip()
        for r in _load_rows()
        if r["enabled"].strip().lower() == "true" and r.get("sector", "").strip()
    }
