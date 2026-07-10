from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from config.settings import settings

# Bridge DeepSwing's env var name to what financedata expects
os.environ.setdefault("ALPHA_VANTAGE_KEY", settings.alpha_vantage_api_key or "")

from financedata import (  # noqa: E402
    get_prices as _get_prices,
    get_prices_batch,
    get_current_price,
    get_vix,
    get_fundamentals as _get_fundamentals,
    ts_to_days as _ts_to_days,
)

logger = logging.getLogger(__name__)

_sector_cache: dict[str, str] = {}


def get_days_to_earnings(tickers: list[str]) -> dict[str, Optional[int]]:
    """
    Days until each ticker's next earnings date (None if unknown or in the past).
    Batched + cached via financedata fundamentals (7-day TTL, parallel fetch).
    """
    if not tickers:
        return {}
    try:
        funds = _get_fundamentals(tickers)
    except Exception as exc:
        logger.warning("Earnings lookup failed: %s", exc)
        return {}
    return {t: _ts_to_days(funds.get(t, {}).get("earnings_timestamp")) for t in tickers}


def fetch_ohlcv(
    ticker: str,
    market: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    return _get_prices(ticker, market=market, period=period, interval=interval)


def _fetch_batch_chunked(tickers: list[str], market: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    chunk_size = max(1, settings.ohlcv_batch_chunk_size)
    results: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        results.update(get_prices_batch(chunk, market=market, period="1y"))
    logger.info("OHLCV batch: %d/%d %s tickers returned", len(results), len(tickers), market)
    return results


def fetch_batch_nordic(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return _fetch_batch_chunked(tickers, market="nordic")


def fetch_batch_eu(tickers: list[str]) -> dict[str, pd.DataFrame]:
    # EU listings use Yahoo suffixes; yfinance path matches US batch fetch.
    return _fetch_batch_chunked(tickers, market="us")


def fetch_batch_us(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return _fetch_batch_chunked(tickers, market="us")


def get_sector(ticker: str) -> str:
    """Universe CSV first (instant, no network), yfinance fallback for US stocks."""
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    from src.data.universe import get_sector_from_universe
    sector = get_sector_from_universe(ticker) or _yf_sector(ticker)
    _sector_cache[ticker] = sector
    return sector


def _yf_sector(ticker: str) -> str:
    import yfinance as yf
    yf_ticker = ticker.replace(".STO", ".ST") if ".STO" in ticker else ticker
    try:
        return yf.Ticker(yf_ticker).info.get("sector") or "Unknown"
    except Exception:
        return "Unknown"
