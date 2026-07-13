from __future__ import annotations

import logging
from typing import Literal, Optional

import dspy

from config.settings import settings
from src.agent.decision import _clamp, build_lm
from src.analysis.screener import ScreenerCandidate
from src.data.options_chain import OptionContract, format_shortlist

logger = logging.getLogger(__name__)

OptionsTrackType = Literal["claude-opt", "gpt-opt"]


def provider_for(track: str) -> Literal["claude", "gpt"]:
    return "claude" if track.startswith("claude") else "gpt"


class OptionTradeDecision(dspy.Signature):
    """
    You are evaluating whether to BUY a single long option on a stock: a CALL when
    the setup is bullish, a PUT when it is bearish. The direction has already been
    chosen by the screener — every contract in option_shortlist matches it (C = call,
    P = put), so your job is judging setup quality, not picking a side. You may ONLY
    choose from the listed contracts — each line has an index, strike, expiry, DTE,
    mid price, delta, theta/day, IV, open interest and spread. Do not invent contracts.

    Return BUY only for a high-conviction setup where the expected underlying move
    clearly exceeds the premium's breakeven within the DTE window — each contract
    line shows its breakeven (BE) and how many times the ATR-projected move covers
    the distance to it. Theta decay is a real, daily cost: prefer contracts whose
    DTE comfortably exceeds the expected swing duration. Check volatility_context
    before buying: when IV is expensive vs realized vol, the move is already priced
    into the premium and you can be right on direction yet still lose — PASS unless
    the setup justifies paying up.

    If action is BUY you must also set the exit plan, as fractions of the premium:
    profit_target_pct / max_loss_pct must be at least 2.0 (e.g. target +0.80 of
    premium against a -0.40 stop). time_stop_dte is the DTE at which the position
    is force-closed regardless of P&L — never ride a long option into its final week.
    """
    technicals: str = dspy.InputField(desc="Technical indicator summary for the underlying stock")
    regime: str = dspy.InputField(desc="Market regime classification and recommended tactics")
    news_summary: str = dspy.InputField(desc="Recent news and sentiment for this ticker")
    macro_context: str = dspy.InputField(desc="Macro economic context relevant to this trade")
    heuristics: str = dspy.InputField(desc="Relevant learned rules from past option trades")
    volatility_context: str = dspy.InputField(
        desc="Realized vol percentile and how expensive ATM IV is vs realized — is the move already priced in?"
    )
    option_shortlist: str = dspy.InputField(desc="Numbered list of purchasable option contracts")

    action: Literal["BUY", "PASS"] = dspy.OutputField(
        desc="BUY to open a long option position (call or put per the shortlist), or PASS to skip this stock"
    )
    contract_index: int = dspy.OutputField(
        desc="Index of the chosen contract from option_shortlist (ignored on PASS)"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence in the decision from 0.0 to 1.0"
    )
    profit_target_pct: float = dspy.OutputField(
        desc="Close at +X of premium, e.g. 0.8 = +80% (must give ratio >= 2.0 vs max_loss_pct)"
    )
    max_loss_pct: float = dspy.OutputField(
        desc="Close at -X of premium, e.g. 0.4 = -40% (between 0.3 and 0.6)"
    )
    time_stop_dte: int = dspy.OutputField(
        desc="Force-close when days-to-expiry falls to this (at least 7)"
    )
    reasoning: str = dspy.OutputField(
        desc="Concise explanation referencing the setup, chosen strike/expiry, and breakeven math"
    )


class OptionsDecisionEngine:
    """Per-track DSPy OptionTradeDecision program; compiled-program lifecycle
    mirrors DecisionEngine (loads compiled/{track}_option_decision.json)."""

    _instances: dict[str, "OptionsDecisionEngine"] = {}

    def __init__(self, track: OptionsTrackType):
        self.track = track
        self._program: Optional[dspy.Predict] = None
        self._lm: Optional[dspy.LM] = None

    @classmethod
    def for_track(cls, track: OptionsTrackType) -> "OptionsDecisionEngine":
        if track not in cls._instances:
            engine = cls(track)
            engine._init_lm()
            engine._load_program()
            cls._instances[track] = engine
        return cls._instances[track]

    def _init_lm(self) -> None:
        if provider_for(self.track) == "claude":
            self._lm = build_lm("claude", settings.claude_decision_model, settings.anthropic_api_key)
        else:
            self._lm = build_lm("gpt", settings.gpt_decision_model, settings.openai_api_key)

    def _load_program(self) -> None:
        compiled_path = settings.compiled_dir / f"{self.track}_option_decision.json"
        self._program = dspy.Predict(OptionTradeDecision)

        if compiled_path.exists():
            try:
                self._program.load(str(compiled_path))
                logger.info("Loaded compiled DSPy option program for %s track from %s", self.track, compiled_path)
            except Exception as exc:
                logger.warning("Failed to load compiled option program for %s: %s — using baseline", self.track, exc)
        else:
            logger.info("No compiled option program found for %s track — using uncompiled baseline", self.track)

    def decide(
        self,
        candidate: ScreenerCandidate,
        shortlist: list[OptionContract],
        news_summary: str,
        macro_context: str,
        heuristics_text: str,
        volatility_context: str = "",
    ) -> Optional[dict]:
        """Run OptionTradeDecision for a candidate + its contract shortlist.
        Returns action/contract/exit-plan dict; invalid contract picks become PASS."""
        if self._program is None or self._lm is None or not shortlist:
            return None

        entry_inputs = {
            "technicals": candidate.signals.to_prompt_str(),
            "regime": candidate.regime.to_prompt_str(),
            "news_summary": news_summary or "No recent news available.",
            "macro_context": macro_context or "No macro data available.",
            "heuristics": heuristics_text or "No relevant heuristics yet.",
            "volatility_context": volatility_context or "No volatility context available.",
            "option_shortlist": format_shortlist(shortlist),
        }

        try:
            with dspy.context(lm=self._lm):
                result = self._program(**entry_inputs)

            action = str(result.action).upper()
            if action not in ("BUY", "PASS"):
                action = "PASS"

            confidence = _clamp(float(result.confidence), 0.0, 1.0)
            reasoning = str(result.reasoning)

            contract: Optional[OptionContract] = None
            profit_target = 0.0
            max_loss = 0.0
            time_stop = 0

            if action == "BUY":
                index = int(result.contract_index)
                if not (0 <= index < len(shortlist)):
                    logger.info(
                        "[%s] %s → BUY with invalid contract_index %d (shortlist size %d) — treating as PASS",
                        self.track, candidate.ticker, index, len(shortlist),
                    )
                    action = "PASS"
                    reasoning = f"(invalid contract index {index}) {reasoning}"
                else:
                    contract = shortlist[index]
                    lo_t, hi_t = settings.options_profit_target_bounds
                    lo_l, hi_l = settings.options_max_loss_bounds
                    profit_target = _clamp(float(result.profit_target_pct), lo_t, hi_t)
                    max_loss = _clamp(float(result.max_loss_pct), lo_l, hi_l)
                    if profit_target < settings.min_rrr * max_loss:
                        profit_target = min(settings.min_rrr * max_loss, hi_t)
                    max_time_stop = max(contract.dte - 1, settings.options_time_stop_min_dte)
                    time_stop = int(_clamp(
                        float(result.time_stop_dte), settings.options_time_stop_min_dte, max_time_stop,
                    ))

            return {
                "action": action,
                "confidence": confidence,
                "contract": contract,
                "profit_target_pct": profit_target,
                "max_loss_pct": max_loss,
                "time_stop_dte": time_stop,
                "reasoning": reasoning,
                "track": self.track,
                "ticker": candidate.ticker,
                "entry_inputs": entry_inputs,
            }

        except Exception as exc:
            logger.error("DSPy option decision error for %s/%s: %s", self.track, candidate.ticker, exc, exc_info=True)
            return None

    def reload(self) -> None:
        self._load_program()
        logger.info("Reloaded DSPy option program for %s track", self.track)


def get_option_decision(
    candidate: ScreenerCandidate,
    track: OptionsTrackType,
    shortlist: list[OptionContract],
    news_summary: str,
    macro_context: str,
    heuristics_text: str,
    volatility_context: str = "",
) -> Optional[dict]:
    """Convenience function — gets or creates track engine and runs decision."""
    engine = OptionsDecisionEngine.for_track(track)
    return engine.decide(candidate, shortlist, news_summary, macro_context, heuristics_text, volatility_context)
