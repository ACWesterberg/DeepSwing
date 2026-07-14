from __future__ import annotations

import logging
import re
from typing import Optional

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# Share-class / listing suffixes that appear in universe names but not headlines
_NAME_SUFFIX_RE = re.compile(r"\s+(a|b|c|sdb|ser\.?\s*[abc])$", re.IGNORECASE)

# Corporate-form words in universe legal names ("Telefonaktiebolaget LM Ericsson
# (publ)", "AB Volvo") that never appear that way in headlines
_CORPORATE_FORM_WORDS = frozenset({
    "ab", "asa", "a/s", "as", "oyj", "abp", "oy", "hf", "plc", "publ",
    "aktiebolag", "aktiebolaget", "telefonaktiebolaget", "aktieselskab",
    "class", "ser", "series", "share", "shares", "sdb", "adr",
    "inc", "corp", "corporation", "ltd", "group", "holding", "holdings",
    "banken",  # 'Skandinaviska Enskilda Banken' — generic, matches any bank headline
})
_PARENTHETICAL_RE = re.compile(r"\([^)]*\)")
_NON_WORD_RE = re.compile(r"[^\w&åäöæøü-]+", re.UNICODE)

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
    Pre-filter articles then call the shared news model (GPT) for per-ticker
    news analysis. Returns a short summary string. Shared across both tracks.
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
        client = openai.OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.gpt_news_model,
            # max_completion_tokens works across the GPT-5 family (reasoning or not);
            # generous headroom so reasoning tokens don't starve the short answer.
            max_completion_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("News analysis error for %s: %s", ticker, exc)
        return "News analysis unavailable."


def _company_name_term(ticker: str) -> str:
    """Lowercased, headline-matchable company name ('' when unknown).
    Headlines say 'Ericsson', never 'ERIC-B' or the universe's legal name
    'Telefonaktiebolaget LM Ericsson (publ)' — so strip parentheticals and
    corporate-form words, then take the last distinctive token (Nordic names
    put the family/brand name last: 'AB Volvo', 'Svenska Handelsbanken AB')."""
    try:
        from src.data.universe import get_name_from_universe
        name = get_name_from_universe(ticker.replace(".STO", ".ST"))
    except Exception:
        return ""
    if not name:
        return ""
    cleaned = _PARENTHETICAL_RE.sub(" ", name)
    cleaned = _NAME_SUFFIX_RE.sub("", cleaned.strip()).lower()
    tokens = [
        t for t in _NON_WORD_RE.split(cleaned)
        if t and t not in _CORPORATE_FORM_WORDS
    ]
    if not tokens:
        return ""
    # Prefer the last token long enough to be distinctive; fall back to the
    # longest one ('Volvo Car' → 'volvo', not the generic 'car').
    for token in reversed(tokens):
        if len(token) >= 4:
            return token
    longest = max(tokens, key=len)
    return longest if len(longest) >= 3 else ""


def _prefilter(ticker: str, articles: list[dict]) -> list[dict]:
    """Keep only articles that mention the company/ticker or key financial terms."""
    ticker_base = ticker.split(".")[0].lower()
    name_term = _company_name_term(ticker)
    relevant = []
    for article in articles:
        text = (article.get("headline") or "").lower()

        if ticker_base in text or (name_term and name_term in text):
            relevant.append(article)
            continue
        if any(kw in text for kw in _RELEVANT_KEYWORDS):
            relevant.append(article)

    return relevant
