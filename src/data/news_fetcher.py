from __future__ import annotations

import logging
import os

from config.settings import settings
from financedata import get_market_headlines, get_news_cached

# Bridge DeepSwing's settings names to the env vars financedata reads.
os.environ.setdefault("NEWS_API_KEY", settings.news_api_key or "")
os.environ.setdefault("FINNHUB_API_KEY", settings.finnhub_api_key or "")
os.environ.setdefault("NEWSAPI_SLOW_THRESHOLD_SECONDS", str(settings.newsapi_slow_threshold_seconds))
os.environ.setdefault("NEWSAPI_COOLDOWN_MINUTES", str(settings.newsapi_cooldown_minutes))

logger = logging.getLogger(__name__)


def fetch_news_for_ticker(ticker: str, market: str, force_refresh: bool = False) -> list[dict]:
    """Recent news for a ticker in financedata's article format:
    [{headline, source_url, published_at, source}].

    Delegates to financedata.get_news_cached, which owns the full source chain
    (RSS → NewsAPI → Finnhub for US when FINNHUB_API_KEY is set → yfinance
    backstop), the NewsAPI rate-limit breaker, and a shared SQLite TTL cache — so
    a ticker fetched by the fund's scan is reused here instead of re-queried. The
    cache TTL mirrors news_refresh_interval_minutes. Pass force_refresh to bypass
    the cache (used by the jump-triggered exit review, where freshness matters)."""
    ttl_hours = max(settings.news_refresh_interval_minutes / 60.0, 0.0)
    results = get_news_cached(
        tickers=[ticker],
        market=market,
        max_age_hours=48,
        ttl_hours=ttl_hours,
        use_fallback=True,
        force_refresh=force_refresh,
    )
    return results.get(ticker, [])


def fetch_market_headlines(market: str = "nordic", max_age_hours: int = 24, limit: int = 20) -> list[dict]:
    """Market-wide (NOT ticker-filtered) macro/geopolitical headlines — the market
    environment signal. Nordic pulls the shared RSS feeds; US uses a broad NewsAPI
    query. Cached per market in the shared DB. Returns
    [{headline, source, published_at}] newest-first."""
    return get_market_headlines(market, max_age_hours=max_age_hours, limit=limit)


def format_market_environment(headlines: list[dict]) -> str:
    """Compact market-wide headline block for the decision/ERL prompts."""
    if not headlines:
        return "No market-wide news available."
    lines = [
        f"- [{h.get('published_at') or '?'} | {h.get('source', '')}] {h['headline']}"
        for h in headlines
    ]
    return "Recent market-wide headlines (newest first):\n" + "\n".join(lines)
