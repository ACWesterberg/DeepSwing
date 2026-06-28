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
)

logger = logging.getLogger(__name__)

_sector_cache: dict[str, str] = {}


def fetch_ohlcv(
    ticker: str,
    market: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    return _get_prices(ticker, market=market, period=period, interval=interval)


def fetch_batch_nordic(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return get_prices_batch(tickers, market="nordic", period="1y")


def fetch_batch_us(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return get_prices_batch(tickers, market="us", period="1y")


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
