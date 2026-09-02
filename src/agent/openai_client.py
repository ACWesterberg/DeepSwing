from __future__ import annotations

import logging

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# Reasoning tokens are billed as output tokens, and the GPT-5 default is
# "medium" — so the three cheap shared calls (triage, news analysis, watch
# classification) were each paying for a thousand-plus invisible tokens to
# produce a ticker list, three sentences, or a one-word verdict. They all want
# the same treatment, so they share one call site.

# Models that reject reasoning_effort outright (a non-reasoning model set via
# env). Remembered per model so one 400 doesn't cost a retry on every call.
_NO_REASONING_SUPPORT: set[str] = set()


def light_completion(model: str, prompt: str, max_completion_tokens: int) -> str:
    """One-shot chat completion for the cheap shared tasks; raises on API failure."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    kwargs: dict = {
        "model": model,
        # max_completion_tokens works across the GPT-5 family (reasoning or not).
        "max_completion_tokens": max_completion_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    effort = settings.gpt_light_reasoning_effort
    send_effort = bool(effort) and model not in _NO_REASONING_SUPPORT
    if send_effort:
        kwargs["reasoning_effort"] = effort

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Matched on the parameter name rather than the SDK's exception class so
        # a rate limit or a timeout re-raises instead of buying a second call.
        if not send_effort or "reasoning_effort" not in str(exc):
            raise
        logger.info("%s rejected reasoning_effort — retrying without it", model)
        _NO_REASONING_SUPPORT.add(model)
        kwargs.pop("reasoning_effort")
        resp = client.chat.completions.create(**kwargs)

    return (resp.choices[0].message.content or "").strip()
