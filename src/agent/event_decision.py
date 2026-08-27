from __future__ import annotations

import logging
from typing import Literal, Optional

import dspy

from config.settings import settings
from src.agent.decision import build_lm, _clamp
from src.analysis.event_model import EventCandidate

logger = logging.getLogger(__name__)


def base_track(track: str) -> str:
    """Map an event track ("claude_events") to the model family it runs on."""
    return "claude" if track.startswith("claude") else "gpt"


class EventTradeDecision(dspy.Signature):
    """
    You are reviewing a statistical edge that has ALREADY been computed on a
    weather prediction-market contract. You are NOT forecasting the weather, and
    you must not try to: the forecast and the fair probability come from the
    National Weather Service and a calibrated error model, both of which are
    better at this than you are.

    Your only job is to judge whether the stated edge is REAL or an ARTIFACT.

    Return PASS when the edge is likely an artifact, for example:
    - The forecast discussion flags unusual uncertainty (a front, a sea breeze, a
      timing question), so the true spread is wider than the model's sigma and the
      edge is overstated.
    - The market has clearly already moved on newer information than the forecast
      the model used.
    - The book is thin or one-sided enough that the ask is not a real price.
    - The contract resolves so soon that the model's uncertainty is implausible.

    Return TRADE only when the edge looks like genuine mispricing: the forecast is
    unremarkable and confident, the market simply has not repriced to it, and the
    quote is real.

    When in doubt, PASS. Most computed edges on liquid weather markets are stale
    or already arbitraged, and passing costs nothing.
    """
    contract_context: str = dspy.InputField(
        desc="Contract terms, model fair probability, market quote, edge, book depth, time to resolution"
    )
    forecast_discussion: str = dspy.InputField(
        desc="The forecast office's plain-language discussion — confidence, fronts, model disagreement"
    )
    heuristics: str = dspy.InputField(desc="Relevant learned rules from past event trades")

    action: Literal["TRADE", "PASS"] = dspy.OutputField(
        desc="TRADE to take the edge, PASS to skip it"
    )
    confidence: float = dspy.OutputField(desc="Confidence in the decision from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(
        desc="Concise explanation citing the specific input that made the edge credible or not"
    )


class EventDecisionEngine:
    """One DSPy EventTradeDecision program per event track."""

    _instances: dict[str, "EventDecisionEngine"] = {}

    def __init__(self, track: str):
        self.track = track
        self._program: Optional[dspy.Predict] = None
        self._lm: Optional[dspy.LM] = None

    @classmethod
    def for_track(cls, track: str) -> "EventDecisionEngine":
        if track not in cls._instances:
            engine = cls(track)
            engine._init_lm()
            engine._load_program()
            cls._instances[track] = engine
        return cls._instances[track]

    def _init_lm(self) -> None:
        if base_track(self.track) == "claude":
            self._lm = build_lm("claude", settings.claude_decision_model, settings.anthropic_api_key)
        else:
            self._lm = build_lm("gpt", settings.gpt_decision_model, settings.openai_api_key)

    def _load_program(self) -> None:
        compiled_path = settings.compiled_dir / f"{self.track}_event_decision.json"
        self._program = dspy.Predict(EventTradeDecision)

        if compiled_path.exists():
            try:
                self._program.load(str(compiled_path))
                logger.info("Loaded compiled event program for %s from %s", self.track, compiled_path)
            except Exception as exc:
                logger.warning("Failed to load compiled event program for %s: %s", self.track, exc)
        else:
            logger.info("No compiled event program for %s — using uncompiled baseline", self.track)

    def decide(
        self,
        candidate: EventCandidate,
        forecast_discussion: str,
        heuristics_text: str,
    ) -> Optional[dict]:
        if self._program is None or self._lm is None:
            logger.error("EventDecisionEngine not initialized for %s", self.track)
            return None

        entry_inputs = {
            "contract_context": candidate.to_prompt_str(),
            "forecast_discussion": forecast_discussion or "No forecast discussion available.",
            "heuristics": heuristics_text or "No relevant heuristics yet.",
        }

        try:
            with dspy.context(lm=self._lm):
                result = self._program(**entry_inputs)

            action = str(result.action).upper()
            if action not in ("TRADE", "PASS"):
                logger.debug("Mapping action '%s' -> PASS for %s/%s", action, self.track, candidate.ticker)
                action = "PASS"

            return {
                "action": action,
                "confidence": _clamp(float(result.confidence), 0.0, 1.0),
                "reasoning": str(result.reasoning),
                "track": self.track,
                "ticker": candidate.ticker,
                "entry_inputs": entry_inputs,
            }
        except Exception as exc:
            logger.error(
                "Event decision error for %s/%s: %s", self.track, candidate.ticker, exc, exc_info=True
            )
            return None

    def reload(self) -> None:
        self._load_program()


def get_event_decision(
    candidate: EventCandidate,
    track: str,
    forecast_discussion: str,
    heuristics_text: str,
) -> Optional[dict]:
    return EventDecisionEngine.for_track(track).decide(
        candidate, forecast_discussion, heuristics_text
    )
