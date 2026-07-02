from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import dspy
from dspy.teleprompt import MIPROv2

if TYPE_CHECKING:
    import pandas as pd

from config.settings import settings
from src.agent.decision import TradeDecision, build_lm
from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import get_portfolio

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]

MIN_TRADES_FOR_OPTIMIZATION = 30

# Scales realized return before the tanh squash. At k=10 a ±10% move lands
# near the (0,1) extremes, which matches typical swing-trade magnitudes.
_PNL_METRIC_SCALE = 10.0


def _pnl_weighted_metric(example, prediction, trace=None) -> float:
    """
    Reward a decision by the money it would have made, not just action-match.

    Each training example carries the realized `pnl_pct` (a fraction) of a trade
    that was actually taken. If the model would BUY, it "earns" that return; if it
    passes, it earns nothing. The result is squashed to (0, 1):

        take a +5% winner  → ~0.73     take a −5% loser → ~0.27
        pass on anything   →  0.50     take a +15% winner → ~0.95

    So passing beats taking a loser but loses to taking a winner, and the *size*
    of each win/loss drives the optimization — which binary action-match ignored.
    """
    pred_action = str(getattr(prediction, "action", "")).upper()
    pnl = float(getattr(example, "pnl_pct", 0.0) or 0.0)
    realized = pnl if pred_action == "BUY" else 0.0
    return 0.5 + 0.5 * math.tanh(realized * _PNL_METRIC_SCALE)


def _make_example(inputs: dict, action: str, pnl_pct: float) -> "dspy.Example":
    return dspy.Example(
        technicals=inputs.get("technicals", ""),
        regime=inputs.get("regime", ""),
        news_summary=inputs.get("news_summary", ""),
        macro_context=inputs.get("macro_context", ""),
        heuristics=inputs.get("heuristics", ""),
        action=action,                # matches BUY/PASS signature
        pnl_pct=float(pnl_pct),       # carried for the P&L-weighted metric
    ).with_inputs("technicals", "regime", "news_summary", "macro_context", "heuristics")


def _label_forward_path(
    window: "pd.DataFrame", price: float, atr: Optional[float]
) -> Optional[tuple[str, float]]:
    """
    Label a PASS decision from the forward OHLC window, simulating the trade
    the system would have taken (ATR stop, min-RRR target, stop-first on a
    both-hit bar). A "missed winner" that would have traded through its stop
    first is a correct PASS, not a missed BUY. Falls back to the horizon-close
    return when ATR wasn't persisted or High/Low aren't available.
    Returns (action, pnl_pct) or None to skip (ambiguous drift).
    """
    has_path = (
        atr is not None and atr > 0
        and "High" in window.columns and "Low" in window.columns
    )
    if has_path:
        stop = price - settings.atr_stop_multiplier * atr
        target = price + settings.min_rrr * (price - stop)
        for _, row in window.iterrows():
            if row["Low"] <= stop:
                return "PASS", stop / price - 1.0     # would have stopped out
            if row["High"] >= target:
                return "BUY", target / price - 1.0    # missed winner

    # No exit hit (or no path data): label from where the horizon closed
    fwd_return = float(window["Close"].dropna().iloc[-1]) / price - 1.0
    if fwd_return >= settings.counterfactual_buy_threshold:
        return "BUY", fwd_return
    if fwd_return <= 0.0:
        return "PASS", fwd_return
    return None  # ambiguous drift — noisy labels help nobody


def _build_counterfactual_examples(track: TrackType, max_examples: int) -> list:
    """
    Label persisted PASS / risk-BLOCKED decisions from what the price actually
    did afterwards. A setup whose simulated forward path hit its target was a
    missed BUY; one that stopped out or went nowhere was a correct skip.
    Without these, the trainset only contains taken trades (survivorship bias)
    and the metric can never penalize passing on winners.
    """
    from datetime import datetime, timedelta

    from src.data.market_data import fetch_ohlcv
    from src.db import Decision, get_session

    horizon = timedelta(days=settings.counterfactual_horizon_days)
    cutoff = datetime.utcnow() - horizon

    session = get_session()
    try:
        rows = (
            session.query(Decision)
            .filter(
                Decision.track == track,
                # BLOCKED = a BUY the risk engine rejected (weak target, cap,
                # correlation…) — never executed, so it labels the same way
                Decision.action.in_(("PASS", "BLOCKED")),
                Decision.entry_inputs.isnot(None),
                Decision.price.isnot(None),
                Decision.timestamp <= cutoff,
            )
            .order_by(Decision.timestamp.desc())
            .limit(max_examples * 3)  # headroom: some get skipped as ambiguous
            .all()
        )
        decisions = [
            {
                "ticker": r.ticker,
                "market": r.market,
                "price": r.price,
                "atr": r.atr,
                "timestamp": r.timestamp,
                "entry_inputs": r.entry_inputs,
            }
            for r in rows
        ]
    finally:
        session.close()

    examples: list = []
    ohlcv_cache: dict[str, object] = {}
    for d in decisions:
        if len(examples) >= max_examples:
            break
        ticker = d["ticker"]
        if ticker not in ohlcv_cache:
            try:
                ohlcv_cache[ticker] = fetch_ohlcv(ticker, d["market"], period="6mo")
            except Exception as exc:
                logger.debug("Counterfactual price fetch failed for %s: %s", ticker, exc)
                ohlcv_cache[ticker] = None
        df = ohlcv_cache[ticker]
        if df is None or df.empty:
            continue

        # Forward window: bars strictly after the decision, up to the horizon.
        # Prices are all native currency, so the return is FX-free.
        start = d["timestamp"].date()
        end = (d["timestamp"] + horizon).date()
        window = df[(df.index.date > start) & (df.index.date <= end)]
        if len(window) < 3 or window["Close"].dropna().empty:
            continue

        labeled = _label_forward_path(window, d["price"], d.get("atr"))
        if labeled is None:
            continue
        label, fwd_return = labeled
        examples.append(_make_example(d["entry_inputs"], label, fwd_return))

    logger.info("MIPRO [%s]: %d counterfactual examples from %d PASS decisions",
                track, len(examples), len(decisions))
    return examples


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

    # Build training examples from closed trades that captured their DSPy inputs
    trainset = []
    for t in trades:
        inputs = getattr(t, "entry_inputs", None)
        if not inputs:
            continue  # Only trades that stored their DSPy inputs can be used
        trainset.append(_make_example(inputs, "BUY" if t.pnl_pct > 0 else "PASS", t.pnl_pct))

    if len(trainset) < 10:
        logger.info("MIPRO [%s]: insufficient labeled examples (%d) — skipping", track, len(trainset))
        return False

    # Augment with counterfactually-labeled PASS decisions, capped at the number
    # of real-trade examples so hindsight labels can't dominate lived outcomes.
    try:
        counterfactuals = _build_counterfactual_examples(
            track, min(settings.counterfactual_max_examples, len(trainset))
        )
        trainset = trainset + counterfactuals
    except Exception as exc:
        logger.warning("MIPRO [%s]: counterfactual build failed (continuing without): %s", track, exc)

    # Split 80/20 — shuffled (fixed seed) so real trades and counterfactuals
    # land in both slices; appending counterfactuals last would otherwise make
    # the val set purely hindsight-labeled.
    import random
    random.Random(42).shuffle(trainset)
    split = int(len(trainset) * 0.8)
    train, val = trainset[:split], trainset[split:]

    # Two roles: the task model runs the program against trades (many calls, so
    # kept on the cheaper decision tier); the prompt model *writes* the candidate
    # instructions (few calls, so given the heaviest reasoner for best prompts).
    # build_lm applies the temperature/max_tokens that GPT-5-class models require.
    if track == "claude":
        task_lm = build_lm(track, settings.claude_decision_model, settings.anthropic_api_key)
        prompt_lm = build_lm(track, settings.claude_prompt_model, settings.anthropic_api_key, max_tokens=4096)
    else:
        task_lm = build_lm(track, settings.gpt_decision_model, settings.openai_api_key)
        prompt_lm = build_lm(track, settings.gpt_prompt_model, settings.openai_api_key, max_tokens=4096)

    program = dspy.Predict(TradeDecision)

    try:
        dspy.configure(lm=task_lm)
        optimizer = MIPROv2(
            metric=_pnl_weighted_metric,
            prompt_model=prompt_lm,  # heavy reasoner writes the instructions
            task_model=task_lm,      # decision-tier model evaluates candidates
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

        # Offsite backup of the new program (best-effort, never fails the run)
        from src.scheduler.backup import backup_compiled_program
        backup_compiled_program(track, metrics.to_dict())

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
