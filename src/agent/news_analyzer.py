from __future__ import annotations

import logging
from typing import Optional

import anthropic

from config.settings import settings

logger = logging.getLogger(__name__)

# Keywords for pre-filtering — only send articles that mention the ticker or these terms
_RELEVANT_KEYWORDS = [
    "earnings", "revenue", "profit", "guidance", "acquisition", "merger",
    "downgrade", "upgrade", "analyst", "insider", "dividend", "recall",
    "lawsuit", "settlement", "partnership", "contract", "ceo", "cfo",
    "vinst", "resultat", "prognos", "förvärv", "analys",  # Swedish
]

_ANALYSIS_PROMPT = """You are analyzing news for a stock in a swing trading context.
Stock: {ticker} | Current price: {price:.4f} | Market: {market}
Technical context: {technicals_brief}

Recent news articles (last 24-48 hours):
{articles}

Provide a concise 2-3 sentence analysis covering:
1. Is there material news that changes the near-term outlook for this stock?
2. Sentiment: bullish / bearish / neutral — and why?
3. Any red flags (earnings miss, downgrade, regulatory issue, insider selling)?

Be specific. If the news is irrelevant to the trade setup, say so briefly."""


def analyze_news(
    ticker: str,
    market: str,
    current_price: float,
    technicals_brief: str,
    articles: list[dict],
) -> str:
    """
    Pre-filter articles then call Claude Haiku for per-ticker news analysis.
    Returns a short summary string.
    """
    relevant = _prefilter(ticker, articles)

    if not relevant:
        return "No recent relevant news found."

    articles_text = "\n\n".join(
        f"[{a.get('source', 'Unknown')} | {a.get('published_at', '?')}]\n"
        f"Headline: {a.get('headline', '')}"
        for a in relevant[:8]  # cap at 8 articles to control tokens
    )

    prompt = _ANALYSIS_PROMPT.format(
        ticker=ticker,
        price=current_price,
        market=market,
        technicals_brief=technicals_brief,
        articles=articles_text,
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.claude_news_model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        logger.error("News analysis error for %s: %s", ticker, exc)
        return "News analysis unavailable."


def _prefilter(ticker: str, articles: list[dict]) -> list[dict]:
    """Keep only articles that mention the ticker symbol or key financial terms."""
    ticker_base = ticker.split(".")[0].lower()
    relevant = []
    for article in articles:
        text = (article.get("headline") or "").lower()

        if ticker_base in text:
            relevant.append(article)
            continue
        if any(kw in text for kw in _RELEVANT_KEYWORDS):
            relevant.append(article)

    return relevant
