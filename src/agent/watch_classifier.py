from __future__ import annotations

import json
import logging
import re

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# One cheap shared call (same pattern as news analysis / triage) turns a batch of
# fresh headlines or an insider-activity change into a directional verdict. The
# watch monitor only pings Telegram on bullish/bearish — neutral stays silent.

_CLASSIFY_PROMPT = """You are screening events on {ticker} for a private investor's alert feed.
They only want to be pinged when an event is clearly directional — routine or ambiguous news must be classified neutral.

Event type: {kind}
{content}

Classify the overall direction these events imply for {ticker} over the coming days/weeks:
- "bullish": clearly positive (beat, upgrade, big contract, insider buying, ...)
- "bearish": clearly negative (miss, downgrade, regulatory trouble, insider selling, ...)
- "neutral": routine, mixed, promotional, or not really about the company

Respond with ONLY a JSON object: {{"verdict": "bullish|bearish|neutral", "reason": "<one short sentence>"}}"""


def classify_watch_event(ticker: str, kind: str, content: str) -> tuple[str, str]:
    """(verdict, reason) for a watchlist event. Fails closed to neutral — a
    classifier outage must never spam the user's phone."""
    prompt = _CLASSIFY_PROMPT.format(ticker=ticker, kind=kind, content=content)
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.watch_classifier_model,
            max_completion_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in reply: {raw[:200]!r}")
        parsed = json.loads(match.group(0))
        verdict = str(parsed.get("verdict", "neutral")).lower()
        if verdict not in ("bullish", "bearish", "neutral"):
            verdict = "neutral"
        return verdict, str(parsed.get("reason", ""))
    except Exception as exc:
        logger.warning("Watch classification failed for %s (%s) — treating as neutral", ticker, exc)
        return "neutral", ""
