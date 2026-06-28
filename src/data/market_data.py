from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf

from config.settings import settings

logger = logging.getLogger(__name__)

# Alpha Vantage free tier: 25 requests/day → cache aggressively and count
_av_cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
AV_CACHE_TTL_HOURS = 4
AV_FREE_TIER_DAILY_LIMIT = 25
_av_daily_count: int = 0
_av_count_date: Optional[date] = None

# Sector cache: ticker → sector string (never expires; sector rarely changes)
_sector_cache: dict[str, str] = {}

# VIX cache: updated at most once per hour
_vix_cache: tuple[Optional[datetime], Optional[float]] = (None, None)
_VIX_CACHE_TTL_HOURS = 1


def fetch_ohlcv(
    ticker: str,
    market: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data. Nordic stocks try Alpha Vantage first, fall back to yfinance."""
    if market == "nordic":
        df = _fetch_alpha_vantage(ticker)
        if df is None or df.empty:
            # Support both legacy .STO and current .ST/.OL/.HE/.CO formats
            yf_ticker = ticker.replace(".STO", ".ST") if ".STO" in ticker else ticker
            df = _fetch_yfinance(yf_ticker, period, interval)
    else:
        df = _fetch_yfinance(ticker, period, interval)

    if df is not None and not df.empty:
        df = _standardize_columns(df)
        df = df.sort_index()
    return df


def _fetch_yfinance(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return None
        return df
    except Exception as exc:
        logger.error("yfinance error for %s: %s", ticker, exc)
        return None


def _fetch_alpha_vantage(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from Alpha Vantage. Results cached for AV_CACHE_TTL_HOURS."""
    global _av_daily_count, _av_count_date

    if not settings.alpha_vantage_api_key:
        return None

    now = datetime.utcnow()
    today = now.date()

    # Reset counter on a new calendar day
    if _av_count_date != today:
        _av_daily_count = 0
        _av_count_date = today

    cached = _av_cache.get(ticker)
    if cached:
        ts, df = cached
        if (now - ts).total_seconds() < AV_CACHE_TTL_HOURS * 3600:
            return df.copy()

    if _av_daily_count >= AV_FREE_TIER_DAILY_LIMIT:
        logger.warning(
            "Alpha Vantage daily limit (%d) reached — falling back to yfinance for %s",
            AV_FREE_TIER_DAILY_LIMIT, ticker,
        )
        return None

    # Strip any Nordic exchange suffix (.STO legacy, .ST, .OL, .HE, .CO, .IC)
    symbol = ticker.split(".")[0]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}"
        f"&outputsize=full"
        f"&apikey={settings.alpha_vantage_api_key}"
    )

    try:
        _av_daily_count += 1
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        ts_key = "Time Series (Daily)"
        if ts_key not in data:
            note = data.get("Note") or data.get("Information", "")
            if "rate limit" in note.lower() or "premium" in note.lower():
                logger.warning("Alpha Vantage rate limit hit for %s", ticker)
            else:
                logger.warning("Alpha Vantage: unexpected response for %s: %s", ticker, note[:100])
            return None

        rows = []
        for date_str, vals in data[ts_key].items():
            rows.append({
                "Date": pd.to_datetime(date_str),
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["5. adjusted close"]),
                "Volume": float(vals["6. volume"]),
            })

        df = pd.DataFrame(rows).set_index("Date").sort_index()
        _av_cache[ticker] = (now, df)
        return df

    except Exception as exc:
        logger.error("Alpha Vantage error for %s: %s", ticker, exc)
        return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            rename[col] = lower.capitalize()
    if rename:
        df = df.rename(columns=rename)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def get_current_price(ticker: str, market: str) -> Optional[float]:
    """Return latest available close price."""
    df = fetch_ohlcv(ticker, market, period="5d", interval="1d")
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def get_vix() -> Optional[float]:
    """Return the latest VIX close. Cached for 1 hour."""
    global _vix_cache
    ts, val = _vix_cache
    if ts and (datetime.utcnow() - ts).total_seconds() < _VIX_CACHE_TTL_HOURS * 3600:
        return val
    try:
        df = yf.Ticker("^VIX").history(period="2d", interval="1d")
        val = float(df["Close"].iloc[-1]) if not df.empty else None
    except Exception as exc:
        logger.warning("VIX fetch error: %s", exc)
        val = None
    _vix_cache = (datetime.utcnow(), val)
    return val


def get_sector(ticker: str) -> str:
    """Return sector for a ticker. Checks universe CSV first, then yfinance (for US stocks)."""
    from src.data.universe import get_sector_from_universe
    cached = _sector_cache.get(ticker)
    if cached:
        return cached
    sector = get_sector_from_universe(ticker)
    if sector is None:
        # Fallback for US stocks not in universe
        yf_ticker = ticker.replace(".STO", ".ST") if ".STO" in ticker else ticker
        try:
            info = yf.Ticker(yf_ticker).info
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
    _sector_cache[ticker] = sector
    return sector


def fetch_batch_nordic(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple Nordic tickers using yfinance batch download.
    Tickers must be in Yahoo Finance format (.ST, .OL, .HE, .CO).
    Alpha Vantage is reserved for individual re-fetches when yfinance fails.
    """
    try:
        raw = yf.download(
            tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance batch download error (Nordic): %s", exc)
        return {}

    results: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        df = _standardize_columns(raw)
        if not df.empty:
            results[tickers[0]] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].dropna(how="all")
                df = _standardize_columns(df)
                if not df.empty:
                    results[ticker] = df
            except Exception:
                pass
    return results


def fetch_batch_us(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple US tickers using yfinance batch download."""
    try:
        raw = yf.download(
            tickers,
            period="1y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance batch download error: %s", exc)
        return {}

    results: dict[str, pd.DataFrame] = {}
    if len(tickers) == 1:
        df = _standardize_columns(raw)
        if not df.empty:
            results[tickers[0]] = df
    else:
        for ticker in tickers:
            try:
                df = raw[ticker].dropna(how="all")
                df = _standardize_columns(df)
                if not df.empty:
                    results[ticker] = df
            except Exception:
                pass
    return results
