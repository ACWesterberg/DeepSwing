from __future__ import annotations

import json
import logging
import re

import openai

from config.settings import settings
from src.analysis.screener import ScreenerCandidate

logger = logging.getLogger(__name__)

# The screener's top-15 all get a news fetch + news analysis + one decision call
# per funded track — the dominant LLM cost of every scan. One cheap shared call
# (same shared-model pattern as news analysis) ranks the screened candidates on
# technicals alone and only the top K proceed to the full pipeline. Both tracks
# see the identical surviving set, so the head-to-head comparison stays fair.

_TRIAGE_PROMPT = """You are pre-screening swing-trade candidates ({market}).
Each line is one candidate that already passed a technical screener, listed best-first by screener score:

{table}

Pick the {keep} setups most worth a full analysis (news review + trade decision).
Judge purely on setup quality: trend alignment, volume confirmation, room to a 2:1 reward/risk target, and regime fit. Prefer diversity of setups over near-duplicates.

Respond with ONLY a JSON array of the chosen tickers, e.g. ["ABC", "XYZ.ST"]."""


def _digest(candidate: ScreenerCandidate, side: str | None) -> str:
    s, r = candidate.signals, candidate.regime
    atr_pct = s.atr_14 / s.current_price * 100 if s.current_price else 0.0
    parts = [
        candidate.ticker,
        f"price {s.current_price:.2f}",
        f"RSI {s.rsi_14:.0f}",
        f"vol {s.volume_ratio:.1f}x avg",
        f"ATR {atr_pct:.1f}%",
        f"{'above' if s.price_above_50sma else 'below'} 50SMA",
        f"bb%B {s.bb_pct_b:.2f}",
        f"{r.regime} (hurst {r.hurst_exponent:.2f})",
    ]
    if side:
        parts.append(f"direction: {side}")
    return " | ".join(parts)


def _call_triage_model(prompt: str) -> str:
    client = openai.OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=settings.triage_model,
        max_completion_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_tickers(raw: str) -> list[str]:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in triage reply: {raw[:200]!r}")
    return [str(t).upper() for t in json.loads(match.group(0)) if isinstance(t, str)]


def triage_candidates(
    candidates: list[ScreenerCandidate],
    market: str,
    sides: dict[str, str] | None = None,
) -> list[ScreenerCandidate]:
    """Cheap shared LLM pass that keeps the top-K screened candidates; the rest
    never reach news fetching or the per-track decision models."""
    keep = settings.triage_keep_top
    if not settings.triage_enabled or keep <= 0 or len(candidates) <= keep:
        return candidates

    table = "\n".join(
        _digest(c, sides.get(c.ticker) if sides else None) for c in candidates
    )
    prompt = _TRIAGE_PROMPT.format(market=market, table=table, keep=keep)

    try:
        chosen_tickers = set(_parse_tickers(_call_triage_model(prompt)))
        chosen = [c for c in candidates if c.ticker.upper() in chosen_tickers][:keep]
        if not chosen:
            raise ValueError("triage reply matched no screened tickers")
        logger.info(
            "Triage kept %d/%d candidates for %s: %s",
            len(chosen), len(candidates), market, ", ".join(c.ticker for c in chosen),
        )
        return chosen
    except Exception as exc:
        # Fail open to the screener's own ranking — a triage outage costs money,
        # never a scan.
        logger.warning(
            "Triage failed for %s (%s) — falling back to screener top %d",
            market, exc, keep,
        )
        return candidates[:keep]
