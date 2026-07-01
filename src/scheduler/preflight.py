from __future__ import annotations

import logging

from config.settings import settings

logger = logging.getLogger(__name__)


def log_model_config() -> None:
    """Log the resolved model IDs for both tracks — always cheap, always runs."""
    logger.info("=== Model configuration ===")
    logger.info(
        "Claude — decision=%s | erl=%s (thinking=%s) | prompt=%s",
        settings.claude_decision_model,
        settings.claude_erl_model,
        settings.claude_erl_extended_thinking,
        settings.claude_prompt_model,
    )
    logger.info(
        "GPT — decision=%s | news=%s (shared) | erl=%s (effort=%s) | prompt=%s",
        settings.gpt_decision_model,
        settings.gpt_news_model,
        settings.gpt_erl_model,
        settings.gpt_erl_reasoning_effort or "off",
        settings.gpt_prompt_model,
    )


def check_models() -> dict[str, bool]:
    """
    Ping each distinct configured model once so a bad model ID or bad credentials
    surfaces at boot rather than at the next scan/ERL/MIPRO run. Never raises;
    returns {provider/model: reachable}. Providers without an API key are skipped.
    """
    results: dict[str, bool] = {}

    anthropic_models = {
        settings.claude_decision_model,
        settings.claude_erl_model,
        settings.claude_prompt_model,
    }
    openai_models = {
        settings.gpt_decision_model,
        settings.gpt_news_model,
        settings.gpt_erl_model,
        settings.gpt_prompt_model,
    }

    if settings.anthropic_api_key:
        for model in sorted(anthropic_models):
            results[f"anthropic/{model}"] = _ping_anthropic(model)
    else:
        logger.warning("Preflight: ANTHROPIC_API_KEY unset — skipping Claude model checks")

    if settings.openai_api_key:
        for model in sorted(openai_models):
            results[f"openai/{model}"] = _ping_openai(model)
    else:
        logger.warning("Preflight: OPENAI_API_KEY unset — skipping GPT model checks")

    bad = [k for k, ok in results.items() if not ok]
    if bad:
        logger.error(
            "Preflight: %d/%d model(s) UNREACHABLE — %s. Fix the ID in .env or check "
            "the provider dashboard; affected track calls will fail until resolved.",
            len(bad), len(results), ", ".join(bad),
        )
    else:
        logger.info("Preflight: all %d configured models reachable", len(results))
    return results


def _ping_anthropic(model: str) -> bool:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "ping"}],
        )
        logger.info("Preflight OK: anthropic/%s", model)
        return True
    except Exception as exc:
        logger.error("Preflight FAIL: anthropic/%s — %s", model, exc)
        return False


def _ping_openai(model: str) -> bool:
    try:
        import openai

        client = openai.OpenAI(api_key=settings.openai_api_key)
        client.chat.completions.create(
            model=model,
            max_completion_tokens=16,
            messages=[{"role": "user", "content": "ping"}],
        )
        logger.info("Preflight OK: openai/%s", model)
        return True
    except Exception as exc:
        logger.error("Preflight FAIL: openai/%s — %s", model, exc)
        return False
