from __future__ import annotations

import json
import logging
from typing import Literal, Optional

import anthropic
import openai

from config.settings import settings
from src.agent.memory import get_store

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]

ERL_PROMPT = """You are analyzing a completed swing trade to extract a reusable trading rule.

Trade summary:
{trade_summary}

Technical signals at entry:
{technicals}

Market regime at entry: {regime}

Outcome: {outcome}

Your task:
1. Identify the PRIMARY cause of this outcome (1-2 sentences).
2. Extract ONE concise trigger-action heuristic in this format:
   Trigger: <specific technical/market condition>
   Action: <what to do or avoid when this trigger fires>
   Quality: <0-10 confidence score>
   Market: <nordic | us | both>
   Regime: <trending | mean-reverting | any>

Focus on generalizable rules that apply beyond this specific stock.
Be specific — vague rules like "check the trend" are useless.
If no clear lesson can be extracted, return Quality: 0."""


def run_erl(
    track: TrackType,
    trade: dict,
    technicals_str: str,
    regime_str: str,
) -> Optional[str]:
    """
    Run Experiential Reflective Learning on a closed trade.
    Extracts a heuristic and saves it to the track's heuristic store.
    Returns the heuristic ID if saved, None otherwise.
    """
    outcome = _describe_outcome(trade)
    trade_summary = (
        f"Ticker: {trade.get('ticker')} | Market: {trade.get('market')}\n"
        f"Entry: {trade.get('entry_price'):.4f} | Exit: {trade.get('exit_price'):.4f}\n"
        f"P&L: {trade.get('pnl_pct', 0)*100:.2f}% | Duration: {trade.get('duration_days', '?')} days\n"
        f"RRR achieved: {trade.get('rrr_achieved', 0):.2f} | Stop hit: {trade.get('stop_hit', False)}"
    )

    prompt = ERL_PROMPT.format(
        trade_summary=trade_summary,
        technicals=technicals_str,
        regime=regime_str,
        outcome=outcome,
    )

    raw_response = _call_model(track, prompt)
    if not raw_response:
        return None

    parsed = _parse_heuristic(raw_response)
    if parsed is None or parsed.get("quality", 0) < 2:
        logger.info("ERL: no useful heuristic extracted for %s trade %s", track, trade.get("id"))
        return None

    store = get_store(track)
    heuristic_id = store.save(
        trigger=parsed["trigger"],
        action=parsed["action"],
        market=parsed.get("market", "both"),
        regime=parsed.get("regime", "any"),
        quality_score=float(parsed["quality"]),
        source_trade_id=trade.get("id"),
    )

    logger.info(
        "ERL: saved heuristic %s for %s track from trade %s (quality=%.1f)",
        heuristic_id[:8], track, trade.get("id"), parsed["quality"],
    )
    return heuristic_id


def _call_model(track: TrackType, prompt: str) -> Optional[str]:
    try:
        if track == "claude":
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            kwargs: dict = {
                "model": settings.claude_erl_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            }
            if settings.claude_erl_extended_thinking:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}
                kwargs["max_tokens"] = 10000
            resp = client.messages.create(**kwargs)
            # With extended thinking, text content is in last text block
            for block in reversed(resp.content):
                if block.type == "text":
                    return block.text
            return None

        else:  # gpt
            client = openai.OpenAI(api_key=settings.openai_api_key)
            kwargs: dict = {
                "model": settings.gpt_erl_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if settings.gpt_erl_reasoning_effort:
                # Reasoning models require max_completion_tokens (not max_tokens)
                # and need headroom for the thinking budget on top of the answer.
                kwargs["reasoning_effort"] = settings.gpt_erl_reasoning_effort
                kwargs["max_completion_tokens"] = 4000
            else:
                kwargs["max_tokens"] = 1024
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content

    except Exception as exc:
        logger.error("ERL model call error for %s track: %s", track, exc)
        return None


def _parse_heuristic(text: str) -> Optional[dict]:
    """Parse the structured heuristic from model response."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Trigger:"):
            result["trigger"] = line[len("Trigger:"):].strip()
        elif line.startswith("Action:"):
            result["action"] = line[len("Action:"):].strip()
        elif line.startswith("Quality:"):
            try:
                result["quality"] = float(line[len("Quality:"):].strip().split()[0])
            except ValueError:
                result["quality"] = 0
        elif line.startswith("Market:"):
            val = line[len("Market:"):].strip().lower()
            result["market"] = val if val in ("nordic", "us", "both") else "both"
        elif line.startswith("Regime:"):
            val = line[len("Regime:"):].strip().lower()
            result["regime"] = val if val in ("trending", "mean-reverting", "any") else "any"

    if "trigger" not in result or "action" not in result:
        logger.debug("ERL: could not parse heuristic from response:\n%s", text[:300])
        return None

    return result


def _describe_outcome(trade: dict) -> str:
    pnl_pct = trade.get("pnl_pct", 0) * 100
    rrr = trade.get("rrr_achieved", 0)
    if pnl_pct > 0:
        return f"PROFITABLE trade: +{pnl_pct:.2f}%, RRR achieved {rrr:.2f}"
    else:
        stop_hit = trade.get("stop_hit", False)
        reason = "stop-loss hit" if stop_hit else "manual/target exit"
        return f"LOSS: {pnl_pct:.2f}% via {reason}, RRR achieved {rrr:.2f}"
