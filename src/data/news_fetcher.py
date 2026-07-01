from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from config.settings import settings
from financedata import fetch_newsapi, get_news, SWEDISH_RSS_FEEDS

os.environ.setdefault("NEWS_API_KEY", settings.news_api_key or "")

logger = logging.getLogger(__name__)

# Market-wide headline cache, keyed by market — shared across both tracks within
# a short window, so a single scan cycle fetches once per market.
_market_cache: dict[str, tuple[datetime, list[dict]]] = {}
_MARKET_TTL_SECONDS = 30 * 60

# Broad market/macro/geopolitical query for US market-wide headlines (NewsAPI).
_US_MARKET_QUERY = (
    '"stock market" OR "Federal Reserve" OR "S&P 500" OR '
    'inflation OR "oil prices" OR geopolitics'
)


def fetch_news_for_ticker(ticker: str, market: str) -> list[dict]:
    """Fetch recent news articles for a ticker. Returns financedata article format:
    [{headline, source_url, published_at, source}]
    """
    feeds = SWEDISH_RSS_FEEDS if market == "nordic" else []
    result = get_news(
        tickers=[ticker],
        feeds=feeds,
        max_age_hours=48,
        use_newsapi=bool(settings.news_api_key),
    )
    return result.get(ticker, [])


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
