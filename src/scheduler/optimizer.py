from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import dspy
from dspy.teleprompt import MIPROv2

from config.settings import settings
from src.agent.decision import TradeDecision
from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import get_portfolio

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]

MIN_TRADES_FOR_OPTIMIZATION = 30


def run_mipro_optimization(track: TrackType) -> bool:
    """
    Run MIPROv2 optimization for a track's DSPy TradeDecision program.
    Requires at least MIN_TRADES_FOR_OPTIMIZATION closed trades.
    Returns True if a new compiled program was saved.
    """
    portfolio = get_portfolio(track)
    trades = portfolio.closed_trades

    if len(trades) < MIN_TRADES_FOR_OPTIMIZATION:
        logger.info(
            "MIPRO [%s]: only %d trades, need %d — skipping",
            track, len(trades), MIN_TRADES_FOR_OPTIMIZATION,
        )
        return False

    logger.info("MIPRO [%s]: starting optimization with %d trades", track, len(trades))

    # Build training examples from closed trades
    trainset = []
    for t in trades:
        if not hasattr(t, "_entry_inputs"):
            continue  # Only trades that stored their DSPy inputs can be used
        example = dspy.Example(
            technicals=t._entry_inputs.get("technicals", ""),
            regime=t._entry_inputs.get("regime", ""),
            news_summary=t._entry_inputs.get("news_summary", ""),
            macro_context=t._entry_inputs.get("macro_context", ""),
            heuristics=t._entry_inputs.get("heuristics", ""),
            action="BUY" if t.pnl_pct > 0 else "HOLD",
        ).with_inputs("technicals", "regime", "news_summary", "macro_context", "heuristics")
        trainset.append(example)

    if len(trainset) < 10:
        logger.info("MIPRO [%s]: insufficient labeled examples (%d) — skipping", track, len(trainset))
        return False

    # Split 80/20
    split = int(len(trainset) * 0.8)
    train, val = trainset[:split], trainset[split:]

    # Configure DSPy LM for this track
    if track == "claude":
        lm = dspy.LM(
            model=f"anthropic/{settings.claude_decision_model}",
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )
    else:
        lm = dspy.LM(
            model=f"openai/{settings.gpt_decision_model}",
            api_key=settings.openai_api_key,
            max_tokens=1024,
        )

    program = dspy.Predict(TradeDecision)

    def metric(example, prediction, trace=None):
        """MIPRO metric: reward correct BUY decisions, penalize false BUYs."""
        pred_action = str(getattr(prediction, "action", "HOLD")).upper()
        true_action = str(example.action).upper()
        if pred_action == true_action:
            return 1.0
        return 0.0

    try:
        dspy.configure(lm=lm)
        optimizer = MIPROv2(
            metric=metric,
            auto="light",  # lighter optimization for Pi resources
            num_threads=1,  # single-threaded for Pi 5
        )
        compiled = optimizer.compile(
            program,
            trainset=train,
            valset=val,
            requires_permission_to_run=False,
        )

        out_path = settings.compiled_dir / f"{track}_trade_decision.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Archive previous
        if out_path.exists():
            from datetime import datetime
            archive = settings.compiled_dir / f"{track}_trade_decision_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            out_path.rename(archive)
            logger.info("MIPRO [%s]: archived previous compiled program to %s", track, archive.name)

        compiled.save(str(out_path))
        logger.info("MIPRO [%s]: saved new compiled program to %s", track, out_path)

        # Reload the engine
        from src.agent.decision import DecisionEngine
        engine = DecisionEngine.for_track(track)
        engine.reload()

        # Log performance metrics
        metrics = compute_metrics(portfolio)
        logger.info(
            "MIPRO [%s]: optimization metric = %.4f (win_rate=%.1f%%, avg_rrr=%.2f)",
            track, metrics.optimization_metric,
            metrics.win_rate * 100, metrics.avg_rrr,
        )

        return True

    except Exception as exc:
        logger.error("MIPRO optimization error for %s track: %s", track, exc, exc_info=True)
        return False


def run_heuristic_refinement(track: TrackType) -> None:
    """Weekly maintenance: prune low-quality heuristics, promote core rules."""
    from src.agent.memory import get_store
    store = get_store(track)
    pruned = store.prune()
    promoted = store.promote_core()
    logger.info(
        "Heuristic refinement [%s]: pruned=%d, promoted_to_core=%d",
        track, pruned, promoted,
    )
