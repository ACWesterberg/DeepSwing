from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Swedish financial RSS feeds
_RSS_FEEDS = [
    ("DI.se", "https://www.di.se/rss"),
    ("Börsdata", "https://borsdata.se/rss"),
    ("Redeye", "https://www.redeye.se/rss"),
]


def fetch_news_for_ticker(ticker: str, market: str) -> list[dict]:
    """Fetch recent news articles for a given ticker."""
    articles: list[dict] = []

    if market == "us" or market == "both":
        articles.extend(_fetch_newsapi(ticker))

    if market == "nordic":
        ticker_base = ticker.split(".")[0]
        articles.extend(_fetch_newsapi(ticker_base))
        articles.extend(_fetch_swedish_rss(ticker_base))

    # Deduplicate by title
    seen: set[str] = set()
    unique: list[dict] = []
    for a in articles:
        title = a.get("title", "")
        if title and title not in seen:
            seen.add(title)
            unique.append(a)

    return unique


def _fetch_newsapi(query: str) -> list[dict]:
    if not settings.news_api_key:
        return []

    since = (datetime.utcnow() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "q": query,
        "from": since,
        "sortBy": "relevancy",
        "language": "en",
        "pageSize": 20,
        "apiKey": settings.news_api_key,
    }

    try:
        resp = httpx.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for item in data.get("articles", []):
            articles.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "published": item.get("publishedAt", ""),
                "url": item.get("url", ""),
            })
        return articles
    except Exception as exc:
        logger.warning("NewsAPI error for '%s': %s", query, exc)
        return []


def _fetch_swedish_rss(ticker_base: str) -> list[dict]:
    articles = []
    for source_name, url in _RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                if ticker_base.lower() in (title + summary).lower():
                    articles.append({
                        "title": title,
                        "description": summary[:400],
                        "source": source_name,
                        "published": getattr(entry, "published", ""),
                        "url": getattr(entry, "link", ""),
                    })
        except Exception as exc:
            logger.debug("RSS fetch error for %s: %s", source_name, exc)
    return articles
