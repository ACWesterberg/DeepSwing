from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Cache for 7 days — index composition rarely changes
_omxs30_cache: tuple[Optional[datetime], list[str]] = (None, [])
_CACHE_TTL_DAYS = 7

# Hardcoded fallback — the 30 constituents as of 2026
_OMXS30_FALLBACK = settings.nordic_watchlist


def get_omxs30_tickers() -> list[str]:
    """
    Return the current OMXS30 constituent list in `.STO` format.
    Fetches from Nasdaq Nordic (weekly cache). Falls back to the
    hardcoded list in settings if the fetch fails.
    """
    global _omxs30_cache
    ts, cached = _omxs30_cache
    if ts and (datetime.utcnow() - ts).total_seconds() < _CACHE_TTL_DAYS * 86400:
        return cached

    tickers = _fetch_nasdaq_nordic() or _fetch_wikipedia()
    if len(tickers) < 20:
        logger.warning("OMXS30 dynamic fetch returned %d tickers — using hardcoded fallback list", len(tickers))
        tickers = list(_OMXS30_FALLBACK)

    _omxs30_cache = (datetime.utcnow(), tickers)
    logger.info("OMXS30 watchlist updated: %d tickers", len(tickers))
    return tickers


def _fetch_nasdaq_nordic() -> list[str]:
    """
    Fetch OMXS30 constituents from the Nasdaq OMX index page.
    The page lists components with their ticker symbols.
    """
    try:
        url = "https://indexes.nasdaqomx.com/Index/Weighting/OMXS30"
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 DeepSwing/1.0"})
        resp.raise_for_status()
        tickers = _parse_nasdaq_nordic_html(resp.text)
        if len(tickers) >= 20:
            return tickers
    except Exception as exc:
        logger.debug("Nasdaq Nordic fetch failed: %s", exc)
    return []


def _parse_nasdaq_nordic_html(html: str) -> list[str]:
    """Extract ticker symbols from the Nasdaq Nordic index weighting page."""
    # The page lists tickers in a table; pattern varies but symbols are uppercase
    # and followed by their exchange (Stockholm listed stocks have "-STO" in the
    # full name or appear as bare symbols).
    # We look for patterns like: ERIC-B, VOLV-B, SAND, HEXA-B
    pattern = re.compile(
        r'<td[^>]*>\s*([A-Z]{2,6}(?:-[A-Z0-9]+)?)\s*</td>',
        re.IGNORECASE,
    )
    candidates = pattern.findall(html)
    # Filter to plausible stock symbols (2-8 chars, not reserved HTML words)
    _skip = {"TD", "TR", "TH", "DIV", "SPAN", "IMG", "A", "P", "UL", "LI", "BR"}
    tickers = []
    seen = set()
    for sym in candidates:
        sym = sym.upper()
        if sym in _skip or len(sym) < 2 or len(sym) > 8:
            continue
        sto = f"{sym}.STO"
        if sto not in seen:
            seen.add(sto)
            tickers.append(sto)
    return tickers[:35]  # OMXS30 has 30 stocks; cap at 35 to allow minor extras


def _fetch_wikipedia() -> list[str]:
    """
    Fallback: fetch OMXS30 composition from the Wikipedia article.
    The article has a table with ticker column that lists the Nasdaq ticker.
    """
    try:
        url = "https://en.wikipedia.org/wiki/OMX_Stockholm_30"
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 DeepSwing/1.0"})
        resp.raise_for_status()
        return _parse_wikipedia_html(resp.text)
    except Exception as exc:
        logger.debug("Wikipedia OMXS30 fetch failed: %s", exc)
    return []


def _parse_wikipedia_html(html: str) -> list[str]:
    """
    Parse the Wikipedia OMXS30 article.
    The wikitable has a 'Ticker' or 'Symbol' column with entries like 'ERIC-B'.
    """
    # Extract the wikitable rows
    table_match = re.search(r'<table class="wikitable[^"]*">(.*?)</table>', html, re.DOTALL)
    if not table_match:
        return []

    table_html = table_match.group(1)
    # Find all <td> values that look like stock tickers
    td_values = re.findall(r'<td[^>]*>\s*<a[^>]*>([^<]+)</a>\s*</td>', table_html)
    td_values += re.findall(r'<td[^>]*>\s*([A-Z]{2,6}(?:[\s-][A-Z0-9]+)?)\s*</td>', table_html)

    tickers = []
    seen: set[str] = set()
    for raw in td_values:
        sym = raw.strip().upper().replace(" ", "-")
        if not re.match(r'^[A-Z]{2,6}(-[A-Z0-9]+)?$', sym):
            continue
        sto = f"{sym}.STO"
        if sto not in seen and len(tickers) < 35:
            seen.add(sto)
            tickers.append(sto)

    return tickers
