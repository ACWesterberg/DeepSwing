from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from config.settings import settings
from financedata import get_news, SWEDISH_RSS_FEEDS

os.environ.setdefault("NEWS_API_KEY", settings.news_api_key or "")

logger = logging.getLogger(__name__)

# Market-wide headline cache — shared across both tracks and both markets within
# a short window, so a single scan cycle fetches the feeds once.
_market_cache: tuple[datetime, list[dict]] | None = None
_MARKET_TTL_SECONDS = 30 * 60


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


def fetch_market_headlines(max_age_hours: int = 24, limit: int = 20) -> list[dict]:
    """
    All recent headlines from the market RSS feeds — NOT filtered to any ticker.
    This is the market-wide / macro / geopolitical environment signal. Cached for
    ~30 min so repeated scans (both markets, both tracks) reuse one fetch.
    Returns [{headline, source, published_at}] newest-first.
    """
    global _market_cache
    import feedparser  # financedata dependency; imported lazily so tests can stub it

    now = datetime.now(timezone.utc)
    if _market_cache and (now - _market_cache[0]).total_seconds() < _MARKET_TTL_SECONDS:
        return _market_cache[1]

    cutoff = now - timedelta(hours=max_age_hours)
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
    items = items[:limit]
    _market_cache = (now, items)
    logger.info("Market-wide news: %d headlines from %d feeds", len(items), len(SWEDISH_RSS_FEEDS))
    return items


def format_market_environment(headlines: list[dict]) -> str:
    """Compact market-wide headline block for the decision/ERL prompts."""
    if not headlines:
        return "No market-wide news available."
    lines = [
        f"- [{h.get('published_at') or '?'} | {h.get('source', '')}] {h['headline']}"
        for h in headlines
    ]
    return "Recent market-wide headlines (newest first):\n" + "\n".join(lines)
