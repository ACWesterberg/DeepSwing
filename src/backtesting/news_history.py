from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from config.settings import BASE_DIR, settings
from src.data.universe import get_name_from_universe

logger = logging.getLogger(__name__)

NEWS_CACHE_DIR = BASE_DIR / "data" / "backtest" / "news"

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_MIN_INTERVAL = 1.0
_FINNHUB_MIN_INTERVAL = 1.1

# Hour (UTC) by which each market's session has closed — the historical "scan
# moment". News published after this on the scan day would not have been known.
_MARKET_CLOSE_UTC = {"nordic": 17, "us": 21}

_GDELT_MARKET_QUERY = {
    "us": '("stock market" OR "Federal Reserve" OR "S&P 500" OR inflation) sourcelang:english',
    "nordic": '(Stockholmsbörsen OR OMXS30 OR Riksbanken OR konjunkturen) sourcelang:swedish',
}

# Trailing share-class tokens on universe names ("Volvo B", "Alfa Laval") that
# would poison a news search for the company.
_SHARE_CLASS_TOKENS = {"a", "b", "c", "d", "sdb", "ser", "pref"}

_last_call_at: dict[str, float] = {}


def company_query_name(ticker: str) -> str:
    """Searchable company name for a ticker: universe name minus share-class suffix."""
    name = get_name_from_universe(ticker) or ticker.split(".")[0]
    tokens = name.split()
    while len(tokens) > 1 and tokens[-1].lower().rstrip(".") in _SHARE_CLASS_TOKENS:
        tokens.pop()
    return " ".join(tokens)


def fetch_ticker_news_asof(
    ticker: str,
    market: str,
    asof: date,
    lookback_hours: int = 48,
    limit: int = 10,
) -> list[dict]:
    """Headlines for a ticker as they existed at `asof` market close, disk-cached.
    Finnhub (US, keyed, ~1y back) preferred; GDELT (free, both markets) fallback."""
    cache_path = NEWS_CACHE_DIR / market / "tickers" / f"{ticker}_{asof.isoformat()}.json"
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    articles: list[dict] = []
    if market == "us" and settings.finnhub_api_key:
        articles = _fetch_finnhub_asof(ticker, asof, lookback_hours, limit)
    if not articles:
        lang = "sourcelang:swedish OR sourcelang:english" if market == "nordic" else "sourcelang:english"
        query = f'"{company_query_name(ticker)}" ({lang})'
        articles = _fetch_gdelt(query, market, asof, lookback_hours, limit)

    _write_cache(cache_path, articles)
    return articles


def fetch_market_headlines_asof(
    market: str, asof: date, max_age_hours: int = 24, limit: int = 20
) -> list[dict]:
    """Market-wide headlines as they existed at `asof` market close (GDELT), disk-cached."""
    cache_path = NEWS_CACHE_DIR / market / "market" / f"{asof.isoformat()}.json"
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached

    articles = _fetch_gdelt(_GDELT_MARKET_QUERY[market], market, asof, max_age_hours, limit)
    _write_cache(cache_path, articles)
    return articles


def format_headlines_block(
    articles: list[dict], asof: date, header: str = "Recent news headlines (newest first):"
) -> str:
    """Compact headline block with relative ages — no absolute dates, so replay
    prompts don't hand the model an anchor for training-data hindsight."""
    if not articles:
        return "No market-wide news available." if "market-wide" in header else "No recent relevant news found."
    lines = [
        f"- [{_relative_age(a.get('published_at'), asof)} | {a.get('source', '')}] {a.get('headline', '')}"
        for a in articles
    ]
    return header + "\n" + "\n".join(lines)


def _relative_age(published_at: Optional[str], asof: date) -> str:
    if not published_at:
        return "recent"
    try:
        pub = datetime.strptime(published_at[:10], "%Y-%m-%d").date()
    except ValueError:
        return "recent"
    days = (asof - pub).days
    if days <= 0:
        return "today"
    return f"{days}d ago"


def _throttle(source: str, min_interval: float) -> None:
    last = _last_call_at.get(source, 0.0)
    wait = min_interval - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    _last_call_at[source] = time.monotonic()


def _fetch_gdelt(
    query: str, market: str, asof: date, lookback_hours: int, limit: int
) -> list[dict]:
    import httpx

    close_utc = datetime(
        asof.year, asof.month, asof.day,
        _MARKET_CLOSE_UTC.get(market, 21), 0, 0, tzinfo=timezone.utc,
    )
    start_utc = close_utc - timedelta(hours=lookback_hours)
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(limit * 2),
        "sort": "DateDesc",
        "startdatetime": start_utc.strftime("%Y%m%d%H%M%S"),
        "enddatetime": close_utc.strftime("%Y%m%d%H%M%S"),
    }

    _throttle("gdelt", _GDELT_MIN_INTERVAL)
    try:
        resp = httpx.get(_GDELT_URL, params=params, timeout=20.0)
        resp.raise_for_status()
        raw = resp.json().get("articles", [])
    except Exception as exc:
        logger.warning("GDELT fetch failed (%s @ %s): %s", query[:40], asof, exc)
        return []

    items: list[dict] = []
    seen: set[str] = set()
    for a in raw:
        title = (a.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        items.append({
            "headline": title[:300],
            "source_url": a.get("url") or "",
            "published_at": _parse_gdelt_date(a.get("seendate")),
            "source": a.get("domain") or "GDELT",
        })
        if len(items) >= limit:
            break
    return items


def _parse_gdelt_date(seendate: Optional[str]) -> str:
    if not seendate:
        return ""
    try:
        dt = datetime.strptime(seendate.rstrip("Z"), "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ""


def _fetch_finnhub_asof(ticker: str, asof: date, lookback_hours: int, limit: int) -> list[dict]:
    import httpx

    _throttle("finnhub", _FINNHUB_MIN_INTERVAL)
    try:
        params = {
            "symbol": ticker,
            "from": (asof - timedelta(hours=lookback_hours)).isoformat(),
            "to": asof.isoformat(),
            "token": settings.finnhub_api_key,
        }
        resp = httpx.get("https://finnhub.io/api/v1/company-news", params=params, timeout=15.0)
        resp.raise_for_status()
        raw = resp.json() or []
    except Exception as exc:
        logger.debug("Finnhub historical news failed for %s @ %s: %s", ticker, asof, exc)
        return []

    close_ts = datetime(
        asof.year, asof.month, asof.day,
        _MARKET_CLOSE_UTC["us"], 0, 0, tzinfo=timezone.utc,
    ).timestamp()

    items: list[dict] = []
    for a in raw:
        title = (a.get("headline") or "").strip()
        ts = a.get("datetime")
        # Finnhub from/to filter is date-granular; drop anything after the close
        if not title or (ts and ts > close_ts):
            continue
        published = ""
        if ts:
            try:
                published = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            except (ValueError, OSError, TypeError):
                published = ""
        items.append({
            "headline": title[:300],
            "source_url": a.get("url") or "",
            "published_at": published,
            "source": a.get("source") or "Finnhub",
        })
        if len(items) >= limit:
            break
    return items


def _read_cache(path: Path) -> Optional[list[dict]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, articles: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.debug("News cache write failed (%s): %s", path.name, exc)
