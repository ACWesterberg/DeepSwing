from __future__ import annotations

import os

from config.settings import settings
from financedata import get_news, SWEDISH_RSS_FEEDS

os.environ.setdefault("NEWS_API_KEY", settings.news_api_key or "")


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
