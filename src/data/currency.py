from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from config.settings import BASE_DIR

logger = logging.getLogger(__name__)

# financedata returns prices in each stock's native quote unit — mapping that to
# a currency is the calling project's job. A suffix fixes the currency for every
# exchange below except London, which quotes most lines in pence and its
# international order book (0XXX.L) in the issuer's home currency.
NORDIC_SUFFIX_CURRENCY: dict[str, str] = {
    ".ST": "SEK",
    ".OL": "NOK",
    ".HE": "EUR",
    ".CO": "DKK",
}

EU_SUFFIX_CURRENCY: dict[str, str] = {
    ".DE": "EUR",
    ".PA": "EUR",
    ".AS": "EUR",
    ".BR": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".SW": "CHF",
    ".WA": "PLN",
    ".VI": "EUR",
    ".LS": "EUR",
    ".IR": "EUR",
}

# Yahoo's minor-unit codes: price ÷ 100 is the major currency. Treating "GBp" as
# GBP booked every pence-quoted entry 100× too high, and a feed that answered in
# pounds on exit then read as a −99% loss.
_SUBUNITS: dict[str, tuple[str, float]] = {
    "GBp": ("GBP", 0.01),
    "GBX": ("GBP", 0.01),
    "ZAc": ("ZAR", 0.01),
    "ILA": ("ILS", 0.01),
}

# Most London lines quote in pence, so an unresolvable .L listing priced
# leniently (an existing position) is assumed to be in pence.
_LONDON_FALLBACK = ("GBP", 0.01)

_CACHE_PATH = BASE_DIR / "data" / "quote_currency.json"
_cache: Optional[dict[str, str]] = None
_cache_lock = threading.Lock()


def normalize_currency(code: str) -> tuple[str, float]:
    """(ISO currency, multiplier to major units) for a Yahoo currency code."""
    return _SUBUNITS.get(code, (code.upper(), 1.0))


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache(cache: dict[str, str]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(cache, sort_keys=True, indent=0))
    except OSError as exc:
        logger.warning("Could not persist quote-currency cache: %s", exc)


def _fetch_yahoo_currency(ticker: str) -> Optional[str]:
    try:
        import yfinance as yf
        code = yf.Ticker(ticker).fast_info.currency
    except Exception as exc:
        logger.warning("Quote-currency lookup failed for %s: %s", ticker, exc)
        return None
    return code if isinstance(code, str) and code else None


def lookup_quote_currency(ticker: str, fetch: bool = True) -> Optional[str]:
    """Yahoo's raw currency code for a listing (e.g. "GBp"), cached on disk forever."""
    with _cache_lock:
        cached = _load_cache().get(ticker)
    if cached or not fetch:
        return cached
    code = _fetch_yahoo_currency(ticker)
    if code:
        with _cache_lock:
            cache = _load_cache()
            cache[ticker] = code
            _save_cache(cache)
    return code


def quote_currency(ticker: str, market: str, strict: bool = False) -> Optional[tuple[str, float]]:
    """(ISO currency, multiplier to major units) a ticker's price is quoted in.

    strict=True returns None instead of guessing when the currency can't be
    established — a wrong assumption corrupts every SEK figure booked for the
    trade. Non-strict keeps a lenient guess so existing positions can still be
    priced and exited consistently with how they were entered."""
    if market == "us":
        if strict and "." in ticker:
            return None
        return ("USD", 1.0)
    if market == "eu" and ticker.endswith(".L"):
        code = lookup_quote_currency(ticker)
        if code:
            return normalize_currency(code)
        return None if strict else _LONDON_FALLBACK
    suffix_map = EU_SUFFIX_CURRENCY if market == "eu" else NORDIC_SUFFIX_CURRENCY
    for suffix, currency in suffix_map.items():
        if ticker.endswith(suffix):
            return (currency, 1.0)
    if market == "eu":
        return None if strict else ("EUR", 1.0)
    if ticker.endswith(".STO"):  # legacy Alpha Vantage suffix — Swedish
        return ("SEK", 1.0)
    return None if strict else ("SEK", 1.0)
