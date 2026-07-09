from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import BASE_DIR

logger = logging.getLogger(__name__)

OHLCV_CACHE_DIR = BASE_DIR / "data" / "backtest" / "ohlcv"
SERIES_CACHE_DIR = BASE_DIR / "data" / "backtest" / "series"

# Calendar days of extra history loaded before the first scan date so SMA200
# (210 trading rows) is warm on day one, holidays included.
WARMUP_CALENDAR_DAYS = 420

_BATCH_SIZE = 50


def load_ohlcv_history(
    tickers: list[str],
    market: str,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """Daily OHLCV for a ticker list over [start, end], disk-cached per ticker."""
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for ticker in tickers:
        cached = _read_cache(_ohlcv_cache_path(market, ticker, start, end))
        if cached is not None:
            result[ticker] = cached
        else:
            missing.append(ticker)

    for i in range(0, len(missing), _BATCH_SIZE):
        batch = missing[i : i + _BATCH_SIZE]
        fetched = _download_batch(batch, market, start, end)
        for ticker, df in fetched.items():
            _write_cache(_ohlcv_cache_path(market, ticker, start, end), df)
            result[ticker] = df

    logger.info(
        "OHLCV history [%s]: %d/%d tickers (%d from cache)",
        market, len(result), len(tickers), len(tickers) - len(missing),
    )
    return result


def load_series_history(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Daily history for a single index/FX symbol (^VIX, ^GSPC, SEK=X...), disk-cached."""
    safe = symbol.replace("^", "_").replace("=", "_")
    path = SERIES_CACHE_DIR / f"{safe}_{start.isoformat()}_{end.isoformat()}.csv"
    cached = _read_cache(path)
    if cached is not None:
        return cached
    fetched = _download_batch([symbol], "series", start, end)
    df = fetched.get(symbol)
    if df is not None:
        _write_cache(path, df)
    return df


def slice_asof(df: pd.DataFrame, day: date) -> pd.DataFrame:
    """Point-in-time view: all bars up to and including `day`."""
    return df[df.index.date <= day]


def asof_close(df: Optional[pd.DataFrame], day: date, strictly_before: bool = False) -> Optional[float]:
    """Last close on or before `day` (strictly before with strictly_before=True)."""
    if df is None or df.empty:
        return None
    sliced = df[df.index.date < day] if strictly_before else df[df.index.date <= day]
    if sliced.empty:
        return None
    return float(sliced["Close"].iloc[-1])


def _download_batch(
    tickers: list[str], market: str, start: date, end: date
) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    from src.backtesting.engine import _standardize

    yf_tickers = [
        t.replace(".STO", ".ST") if market == "nordic" else t for t in tickers
    ]
    ticker_map = dict(zip(yf_tickers, tickers))

    try:
        raw = yf.download(
            yf_tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance history download error: %s", exc)
        return {}
    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if len(yf_tickers) == 1:
        df = _standardize(raw)
        if not df.empty:
            result[tickers[0]] = df
        return result

    for yf_ticker, orig_ticker in ticker_map.items():
        try:
            df = _standardize(raw[yf_ticker].dropna(how="all"))
            if not df.empty:
                result[orig_ticker] = df
        except Exception:
            pass
    return result


def _ohlcv_cache_path(market: str, ticker: str, start: date, end: date) -> Path:
    return OHLCV_CACHE_DIR / market / f"{ticker}_{start.isoformat()}_{end.isoformat()}.csv"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df if not df.empty else None
    except Exception as exc:
        logger.debug("Cache read failed (%s): %s", path.name, exc)
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    except Exception as exc:
        logger.debug("Cache write failed (%s): %s", path.name, exc)
