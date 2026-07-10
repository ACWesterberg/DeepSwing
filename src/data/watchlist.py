from __future__ import annotations

import logging

from config.settings import settings
from src.data.universe import get_eu_tickers as _universe_eu
from src.data.universe import get_nordic_tickers as _universe_nordic
from src.data.universe import get_us_tickers as _universe_us

logger = logging.getLogger(__name__)

# Legacy fallback converted to yfinance format in case universe.csv is missing
_FALLBACK: list[str] = [t.replace(".STO", ".ST") for t in settings.nordic_watchlist]


def get_omxs30_tickers() -> list[str]:
    """
    Return the Nordic watchlist from universe.csv (all enabled main-board stocks:
    OMXS, OSLO, OMXH, OMXC). Tickers are in Yahoo Finance format (.ST/.OL/.HE/.CO).
    Falls back to the hardcoded OMXS30 list if the universe file is unavailable.
    """
    try:
        tickers = _universe_nordic()
        if len(tickers) >= 20:
            logger.debug("Nordic watchlist: %d tickers from universe.csv", len(tickers))
            return tickers
        logger.warning("Universe returned only %d Nordic tickers — using fallback", len(tickers))
    except Exception as exc:
        logger.warning("universe.csv load failed: %s — using hardcoded fallback", exc)
    return _FALLBACK


def get_us_tickers() -> list[str]:
    """
    Return the US watchlist from universe_global.csv (NYSE + NASDAQ enabled stocks).
    Falls back to the hardcoded settings list if the universe file is unavailable.
    """
    try:
        tickers = _universe_us()
        if len(tickers) >= 50:
            logger.debug("US watchlist: %d tickers from universe_global.csv", len(tickers))
            return tickers
        logger.warning("Universe returned only %d US tickers — using fallback", len(tickers))
    except Exception as exc:
        logger.warning("universe_global.csv load failed: %s — using hardcoded fallback", exc)
    return settings.us_watchlist


def get_eu_watchlist() -> list[str]:
    """
    Return the continental-EU watchlist from universe_eu.csv (LSE, XETRA,
    Euronext, SIX, etc.). Falls back to an empty list if unavailable.
    """
    try:
        tickers = _universe_eu()
        if len(tickers) >= 50:
            logger.debug("EU watchlist: %d tickers from universe_eu.csv", len(tickers))
            return tickers
        logger.warning("Universe returned only %d EU tickers", len(tickers))
    except Exception as exc:
        logger.warning("universe_eu.csv load failed: %s", exc)
    return []
