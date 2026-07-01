from __future__ import annotations

import json
import logging
from typing import Literal, Optional

import dspy

from config.settings import settings
from src.analysis.screener import ScreenerCandidate

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]
ActionType = Literal["BUY", "SELL", "HOLD"]


class TradeDecision(dspy.Signature):
    """
    Analyze a stock setup as an experienced swing trader and decide whether to buy, sell, or hold.
    You must justify every output with specific references to the provided data.
    Return HOLD unless there is a high-conviction setup with clear risk/reward.

    If action is BUY, you MUST ensure the risk/reward ratio (RRR) is at least 2.0:
      RRR = (target - entry) / (entry - stop_loss) >= 2.0
    Example: entry=100, stop_loss=95 (risk=5) → target must be >= 110 (reward >= 10).
    A BUY with RRR below 2.0 will be automatically rejected — so set a wide enough target.
    """
    technicals: str = dspy.InputField(desc="Technical indicator summary for the stock")
    regime: str = dspy.InputField(desc="Market regime classification and recommended tactics")
    news_summary: str = dspy.InputField(desc="Recent news and sentiment for this ticker")
    macro_context: str = dspy.InputField(desc="Macro economic context relevant to this trade")
    heuristics: str = dspy.InputField(desc="Relevant learned rules from past trades")

    action: Literal["BUY", "SELL", "HOLD"] = dspy.OutputField(
        desc="Trading decision: BUY (enter long), SELL (exit/short), or HOLD (no action)"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the decision from 0.0 (no confidence) to 1.0 (maximum confidence)"
    )
    stop_loss: float = dspy.OutputField(
        desc="Suggested stop-loss price level (must be below entry for BUY)"
    )
    target: float = dspy.OutputField(
        desc="Price target for the trade (must give RRR >= 2.0)"
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation of why this action was chosen, referencing specific signals"
    )


class DecisionEngine:
    """
    Wraps a DSPy TradeDecision program per track.
    Loads compiled program from disk if available, otherwise uses uncompiled baseline.
    """

    _instances: dict[str, "DecisionEngine"] = {}

    def __init__(self, track: TrackType):
        self.track = track
        self._program: Optional[dspy.Predict] = None
        self._lm: Optional[dspy.LM] = None

    @classmethod
    def for_track(cls, track: TrackType) -> "DecisionEngine":
        if track not in cls._instances:
            engine = cls(track)
            engine._init_lm()
            engine._load_program()
            cls._instances[track] = engine
        return cls._instances[track]

    def _init_lm(self) -> None:
        if self.track == "claude":
            self._lm = dspy.LM(
                model=f"anthropic/{settings.claude_decision_model}",
                api_key=settings.anthropic_api_key,
                max_tokens=1024,
            )
        else:
            self._lm = dspy.LM(
                model=f"openai/{settings.gpt_decision_model}",
                api_key=settings.openai_api_key,
                max_tokens=1024,
            )

    def _load_program(self) -> None:
        compiled_path = settings.compiled_dir / f"{self.track}_trade_decision.json"
        self._program = dspy.Predict(TradeDecision)

        if compiled_path.exists():
            try:
                self._program.load(str(compiled_path))
                logger.info("Loaded compiled DSPy program for %s track from %s", self.track, compiled_path)
            except Exception as exc:
                logger.warning("Failed to load compiled program for %s: %s — using baseline", self.track, exc)
        else:
            logger.info("No compiled program found for %s track — using uncompiled baseline", self.track)

    def decide(
        self,
        candidate: ScreenerCandidate,
        news_summary: str,
        macro_context: str,
        heuristics_text: str,
    ) -> Optional[dict]:
        """
        Run the DSPy TradeDecision program for a candidate.
        Returns a dict with action, confidence, stop_loss, target, reasoning.
        """
        if self._program is None or self._lm is None:
            logger.error("DecisionEngine not initialized for track %s", self.track)
            return None

        # Capture the exact inputs fed to the program so MIPRO can train on them later
        entry_inputs = {
            "technicals": candidate.signals.to_prompt_str(),
            "regime": candidate.regime.to_prompt_str(),
            "news_summary": news_summary or "No recent news available.",
            "macro_context": macro_context or "No macro data available.",
            "heuristics": heuristics_text or "No relevant heuristics yet.",
        }

        try:
            with dspy.context(lm=self._lm):
                result = self._program(**entry_inputs)

            action = str(result.action).upper()
            if action not in ("BUY", "SELL", "HOLD"):
                logger.warning("Invalid action '%s' from %s track — defaulting to HOLD", action, self.track)
                action = "HOLD"

            confidence = _clamp(float(result.confidence), 0.0, 1.0)
            stop_loss = float(result.stop_loss)
            target = float(result.target)
            action, target = _fix_rrr(action, candidate.signals.current_price, stop_loss, target, settings.min_rrr)

            return {
                "action": action,
                "confidence": confidence,
                "stop_loss": stop_loss,
                "target": target,
                "reasoning": str(result.reasoning),
                "track": self.track,
                "ticker": candidate.ticker,
                "entry_inputs": entry_inputs,
            }

        except Exception as exc:
            logger.error("DSPy decision error for %s/%s: %s", self.track, candidate.ticker, exc, exc_info=True)
            return None

    def reload(self) -> None:
        """Reload compiled program from disk (called after MIPRO optimization)."""
        self._load_program()
        logger.info("Reloaded DSPy program for %s track", self.track)


def get_decision(
    candidate: ScreenerCandidate,
    track: TrackType,
    news_summary: str,
    macro_context: str,
    heuristics_text: str,
) -> Optional[dict]:
    """Convenience function — gets or creates track engine and runs decision."""
    engine = DecisionEngine.for_track(track)
    return engine.decide(candidate, news_summary, macro_context, heuristics_text)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _fix_rrr(
    action: str,
    entry: float,
    stop_loss: float,
    target: float,
    min_rrr: float,
) -> tuple[str, float]:
    """
    If a BUY decision has RRR between 1.0 and min_rrr, stretch the target to meet
    the minimum. If RRR < 1.0 (target barely above or below stop), leave it as-is
    so the risk validator rejects it — the stop placement itself is bad.
    Returns (action, corrected_target).
    """
    if action != "BUY":
        return action, target
    risk = entry - stop_loss
    if risk <= 0:
        return action, target
    rrr = (target - entry) / risk
    if rrr < 1.0:
        return action, target  # bad stop placement — let risk validator reject
    if rrr < min_rrr:
        corrected = entry + min_rrr * risk
        logger.debug(
            "Stretched target from %.4f to %.4f to meet RRR %.1f (was %.2f)",
            target, corrected, min_rrr, rrr,
        )
        return action, corrected
    return action, target
