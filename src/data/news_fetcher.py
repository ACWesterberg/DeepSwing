from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from config.settings import settings
from financedata import fetch_newsapi, get_news, SWEDISH_RSS_FEEDS

os.environ.setdefault("NEWS_API_KEY", settings.news_api_key or "")

logger = logging.getLogger(__name__)

# Market-wide headline cache, keyed by market — shared across both tracks within
# a short window, so a single scan cycle fetches once per market.
_market_cache: dict[str, tuple[datetime, list[dict]]] = {}
_MARKET_TTL_SECONDS = 30 * 60

# Per-ticker news cache, keyed by (ticker, market). Candidates recur across the
# 15-min scans, so this avoids re-hitting NewsAPI for the same ticker.
_ticker_cache: dict[tuple[str, str], tuple[datetime, list[dict]]] = {}

# Rate-limit breaker: when a NewsAPI fetch stalls on 429 backoff we skip NewsAPI
# (RSS only) until this time, so one throttled ticker doesn't cost ~1 min each.
_newsapi_cooldown_until: Optional[datetime] = None


def _newsapi_available() -> bool:
    if not settings.news_api_key:
        return False
    if _newsapi_cooldown_until and datetime.now(timezone.utc) < _newsapi_cooldown_until:
        return False
    return True


# Broad market/macro/geopolitical query for US market-wide headlines (NewsAPI).
_US_MARKET_QUERY = (
    '"stock market" OR "Federal Reserve" OR "S&P 500" OR '
    'inflation OR "oil prices" OR geopolitics'
)


def fetch_news_for_ticker(ticker: str, market: str, force_refresh: bool = False) -> list[dict]:
    """Fetch recent news articles for a ticker. Returns financedata article format:
    [{headline, source_url, published_at, source}]

    Results are cached per (ticker, market) for news_refresh_interval_minutes.
    If NewsAPI is being rate-limited, it's skipped (RSS only) until the cooldown
    expires so a throttled ticker doesn't stall the scan. Pass force_refresh to
    bypass the cache (used by the jump-triggered exit review, where freshness
    matters most)."""
    global _newsapi_cooldown_until

    now = datetime.now(timezone.utc)
    key = (ticker, market)
    ttl = settings.news_refresh_interval_minutes * 60
    cached = _ticker_cache.get(key)
    if not force_refresh and cached and (now - cached[0]).total_seconds() < ttl:
        return cached[1]

    feeds = SWEDISH_RSS_FEEDS if market == "nordic" else []
    use_newsapi = _newsapi_available()
    started = time.monotonic()
    result = get_news(
        tickers=[ticker],
        feeds=feeds,
        max_age_hours=48,
        use_newsapi=use_newsapi,
    )
    elapsed = time.monotonic() - started
    articles = result.get(ticker, [])

    # A slow NewsAPI call means we hit 429 backoff — trip the breaker so the rest
    # of this scan skips NewsAPI and just uses RSS.
    threshold = settings.newsapi_slow_threshold_seconds
    if use_newsapi and threshold > 0 and elapsed >= threshold:
        _newsapi_cooldown_until = now + timedelta(minutes=settings.newsapi_cooldown_minutes)
        logger.warning(
            "NewsAPI stalled %.0fs on %s — skipping NewsAPI for %d min (RSS only)",
            elapsed, ticker, settings.newsapi_cooldown_minutes,
        )

    # When the primary (NewsAPI/RSS) yields nothing — common for US, which has no
    # RSS feeds, and while the breaker is tripped — fall back to a free per-ticker
    # source so US tickers still get news.
    if not articles:
        fallback = _fetch_fallback_news(ticker, market)
        if fallback:
            logger.info("News fallback for %s: %d article(s)", ticker, len(fallback))
            articles = fallback

    _ticker_cache[key] = (now, articles)
    return articles


def _fetch_fallback_news(ticker: str, market: str) -> list[dict]:
    """Free per-ticker news when the primary source is empty. Finnhub is preferred
    for US when a key is configured; yfinance (Yahoo) is the universal free
    backstop for both markets."""
    if market == "us" and settings.finnhub_api_key:
        articles = _fetch_finnhub_news(ticker)
        if articles:
            return articles
    return _fetch_yfinance_news(ticker)


def _fetch_yfinance_news(ticker: str, limit: int = 10) -> list[dict]:
    """Per-ticker news via yfinance (Yahoo). No API key. Handles both the newer
    ('content' envelope) and older (flat) yfinance news schemas."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:
        logger.debug("yfinance news failed for %s: %s", ticker, exc)
        return []

    items: list[dict] = []
    for entry in raw[:limit]:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if isinstance(content, dict):  # newer schema (yfinance >= ~0.2.40)
            title = (content.get("title") or "").strip()
            source = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
            url = ((content.get("canonicalUrl") or {}).get("url")
                   or (content.get("clickThroughUrl") or {}).get("url") or "")
            published = str(content.get("pubDate") or content.get("displayTime") or "")
            published = published.replace("T", " ").rstrip("Z")[:16]
        else:  # older flat schema
            title = (entry.get("title") or "").strip()
            source = entry.get("publisher") or "Yahoo Finance"
            url = entry.get("link") or ""
            ts = entry.get("providerPublishTime")
            published = ""
            if ts:
                try:
                    published = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                except (ValueError, OSError, TypeError):
                    published = ""
        if not title:
            continue
        items.append({
            "headline": title[:300],
            "source_url": url,
            "published_at": published,
            "source": source,
        })
    return items


def _fetch_finnhub_news(ticker: str, limit: int = 10) -> list[dict]:
    """Per-ticker company news via Finnhub (needs finnhub_api_key). Dormant until
    a key is set, so this is the drop-in 'preferred US source' for later."""
    if not settings.finnhub_api_key:
        return []
    try:
        import httpx
        today = datetime.now(timezone.utc).date()
        params = {
            "symbol": ticker,
            "from": (today - timedelta(days=7)).isoformat(),
            "to": today.isoformat(),
            "token": settings.finnhub_api_key,
        }
        resp = httpx.get("https://finnhub.io/api/v1/company-news", params=params, timeout=10.0)
        resp.raise_for_status()
        raw = resp.json() or []
    except Exception as exc:
        logger.debug("Finnhub news failed for %s: %s", ticker, exc)
        return []

    items: list[dict] = []
    for a in raw[:limit]:
        title = (a.get("headline") or "").strip()
        if not title:
            continue
        published = ""
        ts = a.get("datetime")
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
    return items


def fetch_market_headlines(market: str = "nordic", max_age_hours: int = 24, limit: int = 20) -> list[dict]:
    """
    Recent market-wide headlines — NOT filtered to any ticker. The market-wide /
    macro / geopolitical environment signal. Nordic pulls the RSS feeds directly;
    US uses a broad NewsAPI query. Cached ~30 min per market so a scan cycle
    fetches once. Returns [{headline, source, published_at}] newest-first.
    """
    now = datetime.now(timezone.utc)
    cached = _market_cache.get(market)
    if cached and (now - cached[0]).total_seconds() < _MARKET_TTL_SECONDS:
        return cached[1]

    if market == "us":
        items = _fetch_us_market_headlines(max_age_hours, limit)
    else:
        items = _fetch_rss_market_headlines(max_age_hours, limit)

    _market_cache[market] = (now, items)
    logger.info("Market-wide news [%s]: %d headlines", market, len(items))
    return items


def _fetch_rss_market_headlines(max_age_hours: int, limit: int) -> list[dict]:
    import feedparser  # financedata dependency; imported lazily so tests can stub it

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    items: list[dict] = []
    seen: set[str] = set()

    for feed_url in SWEDISH_RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            logger.debug("Market RSS error (%s): %s", feed_url, exc)
            continue
        source = (getattr(feed.feed, "title", "") or feed_url.split("/")[2])[:30]
        for entry in getattr(feed, "entries", []):
            title = (getattr(entry, "title", "") or "").strip()
            if not title or title.lower() in seen:
                continue
            pub_dt = None
            pub_str = None
            for attr in ("published_parsed", "updated_parsed"):
                parsed = getattr(entry, attr, None)
                if parsed:
                    pub_dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                    pub_str = pub_dt.strftime("%Y-%m-%d %H:%M")
                    break
            if pub_dt and pub_dt < cutoff:
                continue
            seen.add(title.lower())
            items.append({"headline": title[:300], "source": source, "published_at": pub_str})

    items.sort(key=lambda a: a.get("published_at") or "", reverse=True)
    return items[:limit]


def _fetch_us_market_headlines(max_age_hours: int, limit: int) -> list[dict]:
    if not settings.news_api_key:
        logger.info("US market news skipped — no NEWS_API_KEY configured")
        return []
    try:
        articles = fetch_newsapi(_US_MARKET_QUERY, max_age_hours=max_age_hours, page_size=limit)
    except Exception as exc:
        logger.warning("US market news fetch failed: %s", exc)
        return []

    items: list[dict] = []
    seen: set[str] = set()
    for a in articles:
        title = (a.get("headline") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        items.append({
            "headline": title[:300],
            "source": a.get("source", "NewsAPI"),
            "published_at": (a.get("published_at") or "").replace("T", " ").rstrip("Z")[:16],
        })
    items.sort(key=lambda a: a.get("published_at") or "", reverse=True)
    return items[:limit]


def format_market_environment(headlines: list[dict]) -> str:
    """Compact market-wide headline block for the decision/ERL prompts."""
    if not headlines:
        return "No market-wide news available."
    lines = [
        f"- [{h.get('published_at') or '?'} | {h.get('source', '')}] {h['headline']}"
        for h in headlines
    ]
    return "Recent market-wide headlines (newest first):\n" + "\n".join(lines)
