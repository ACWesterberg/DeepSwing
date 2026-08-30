"""
MIPROv2 optimization of the DSPy decision program.

`run_mipro_optimization` trains on lived trades plus counterfactually-labelled
PASS/BLOCKED decisions, so the trainset covers both sides of the decision.
Labelling a skipped setup only needs the underlying's forward path, which
`_label_forward_path` simulates against the stop and target the system would
have used.
"""
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
from src.agent.compiled_program import BASELINE, program_fingerprint
from src.agent.decision import TradeDecision, build_lm
from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import breakeven_from_costs, get_portfolio

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]

MIN_TRADES_FOR_OPTIMIZATION = 30

# Real (lived) examples required before counterfactuals are allowed to top up
# the trainset. Keeps a compile from being decided almost entirely by
# hindsight-labelled setups that were never actually traded.
MIN_REAL_EXAMPLES = 10

# Total labelled examples required before MIPRO runs at all, counted AFTER
# counterfactual augmentation. MIPRO holds out 20% and picks the winning
# instructions on that slice, so the old floor of 10 meant selecting between
# candidate instruction sets on two examples. Against swing-trade P&L dispersion
# the best of a dozen candidates beats the field by more than the real spread
# between them from chance alone, and the compiled program would look like an
# improvement while being sampling error. This is the "should we believe it"
# threshold, not the "can it run" one.
MIN_EXAMPLES_FOR_OPTIMIZATION = 25

# Scales the realized R-multiple before the tanh squash. Chosen so the metric
# still separates outcomes across the range a swing book actually produces
# (-1R to +5R): a 2.5R and a 5R differ by 0.12 here, where the previous
# formulation — raw pnl_pct at k=10 — put them 0.045 apart and treated a +15%
# and a +30% trade as near-identical. Optimizing for a fatter right tail
# requires a metric that can see one.
_R_METRIC_SCALE = 0.35


def _pnl_weighted_metric(example, prediction, trace=None) -> float:
    """
    Reward a decision by the money it would have made, not just action-match.

    Each training example carries the realized R-multiple of the trade — profit
    per unit of risk taken, the unit the rest of the system already thinks in.
    If the model would BUY it "earns" that R; if it passes it earns nothing.
    Squashed to (0, 1):

        take a +1R winner  → ~0.67     take a -1R loser → ~0.33
        pass on anything   →  0.50     take a +5R winner → ~0.97

    Scoring R rather than raw return matters because position size is already
    risk-normalized: a 2% move on a tight stop and a 10% move on a wide one are
    the same trade to the book, and a metric denominated in percent would
    reward the volatile one for volatility alone.

    Note passing always scores exactly 0.5. On a trainset of mostly losers the
    do-nothing program therefore wins, and only the counterfactual "missed BUY"
    examples pull against that — which is why they are not optional.
    """
    pred_action = str(getattr(prediction, "action", "")).upper()
    r = float(getattr(example, "r_multiple", 0.0) or 0.0)
    realized = r if pred_action == "BUY" else 0.0
    return 0.5 + 0.5 * math.tanh(realized * _R_METRIC_SCALE)


def _make_example(inputs: dict, action: str, r_multiple: float) -> "dspy.Example":
    return dspy.Example(
        technicals=inputs.get("technicals", ""),
        regime=inputs.get("regime", ""),
        news_summary=inputs.get("news_summary", ""),
        macro_context=inputs.get("macro_context", ""),
        heuristics=inputs.get("heuristics", ""),
        action=action,                    # matches BUY/PASS signature
        r_multiple=float(r_multiple),     # carried for the R-weighted metric
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
    Returns (action, r_multiple) or None to skip (ambiguous drift, or a risk
    denominator that can't be established). R rather than a raw return so
    counterfactual examples are denominated the same way lived trades are.
    """
    # Two separate requirements: ATR gives the risk denominator that makes an
    # R-multiple meaningful at all; High/Low additionally allow simulating the
    # intraday path. Without High/Low we can still label from the close.
    has_atr = atr is not None and atr > 0
    has_path = has_atr and "High" in window.columns and "Low" in window.columns
    if has_path:
        stop = price - settings.atr_stop_multiplier * atr
        risk = price - stop
        target = price + settings.min_rrr * risk
        arm_at = price + settings.breakeven_arm_atr_multiplier * atr
        floor = breakeven_from_costs(price, settings.commission_pct, settings.simulated_slippage)
        armed = False
        for _, row in window.iterrows():
            # Mirror the live exit policy, or hindsight examples carry a clean
            # +min_rrr while real trades of the same setup carry a floored or
            # trailed result — the metric would then reward BUY on the
            # counterfactual half of the trainset for free.
            effective_stop = max(stop, floor) if armed else stop
            if row["Low"] <= effective_stop:
                return "PASS", (effective_stop - price) / risk
            if row["High"] >= target:
                return "BUY", (target - price) / risk   # missed winner
            # Arm from the close after the exit checks — within a bar the
            # order of high and low is unknown (same rule as the backtester).
            if not armed and settings.breakeven_arm_atr_multiplier > 0 and row["Close"] >= arm_at:
                armed = True

    # No exit hit, or no High/Low to walk. Without ATR there is no risk to
    # divide by, and inventing a denominator would hand hindsight examples a
    # risk basis the lived half never got — the same bias the floor mirror
    # above removes. PASS decisions have persisted their ATR since the
    # counterfactual pipeline shipped, so this only skips pre-upgrade rows.
    if not has_atr:
        return None
    risk_frac = settings.atr_stop_multiplier * atr / price
    fwd_return = float(window["Close"].dropna().iloc[-1]) / price - 1.0
    if fwd_return >= settings.counterfactual_buy_threshold:
        return "BUY", fwd_return / risk_frac
    if fwd_return <= 0.0:
        return "PASS", fwd_return / risk_frac
    return None  # ambiguous drift — noisy labels help nobody


def score_heuristics_from_decisions(track: TrackType, max_decisions: int = 200) -> int:
    """
    Re-score heuristics against the setups they talked the model out of.

    `record_outcome` only ever ran on closed trades, so a heuristic was judged
    on the subset of its influence that produced a position. A rule that argued
    for passing was never credited when passing was right, nor charged when it
    cost a winner — the same survivorship bias the counterfactual trainset was
    built to remove, one layer down in the thing that decides which rules reach
    the prompt at all.

    Sign follows the decision the heuristics informed, not the price move:
    a PASS is right when the forward path went nowhere, a BLOCKED BUY was the
    model wanting in, so it is scored like a taken trade.

    Returns the number of decisions scored. Idempotent via `heuristics_scored`,
    since `record_outcome` has no idempotency of its own and would double-count.
    """
    from datetime import datetime, timedelta

    from src.agent.memory import get_store
    from src.data.market_data import fetch_ohlcv
    from src.db import Decision, get_session

    horizon = timedelta(days=settings.counterfactual_horizon_days)
    cutoff = datetime.utcnow() - horizon
    store = get_store(track)

    session = get_session()
    scored = 0
    try:
        rows = (
            session.query(Decision)
            .filter(
                Decision.track == track,
                Decision.action.in_(("PASS", "BLOCKED")),
                Decision.entry_inputs.isnot(None),
                Decision.price.isnot(None),
                Decision.timestamp <= cutoff,
                Decision.heuristics_scored.isnot(True),
            )
            .order_by(Decision.timestamp.desc())
            .limit(max_decisions)
            .all()
        )

        ohlcv_cache: dict[str, object] = {}
        for row in rows:
            ids = (row.entry_inputs or {}).get("heuristic_ids") or []
            if not ids:
                # Predates heuristic_ids on skipped setups; nothing to score,
                # but mark it so it isn't re-fetched on every weekly run.
                row.heuristics_scored = True
                continue

            ticker = row.ticker
            if ticker not in ohlcv_cache:
                try:
                    ohlcv_cache[ticker] = fetch_ohlcv(ticker, row.market, period="6mo")
                except Exception as exc:
                    logger.debug("Heuristic scoring fetch failed for %s: %s", ticker, exc)
                    ohlcv_cache[ticker] = None
            df = ohlcv_cache[ticker]
            if df is None or df.empty:
                continue

            start = row.timestamp.date()
            end = (row.timestamp + horizon).date()
            window = df[(df.index.date > start) & (df.index.date <= end)]
            if len(window) < 3 or window["Close"].dropna().empty:
                continue

            labeled = _label_forward_path(window, row.price, row.atr)
            if labeled is None:
                continue  # ambiguous drift — leave unscored, it may resolve
            _, fwd_return = labeled

            # A PASS is vindicated by a move that did not pay, so the signal is
            # the negation of the forward return. A BLOCKED BUY carried the
            # model's intent to buy, so it scores in the same direction a taken
            # trade would have.
            signal = fwd_return if row.action == "BLOCKED" else -fwd_return
            store.record_outcome(ids, signal)
            row.heuristics_scored = True
            scored += 1

        session.commit()
    finally:
        session.close()

    if scored:
        logger.info("Scored %d skipped setup(s) against %s heuristics", scored, track)
    return scored


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
    # Queried newest-first (to take the freshest N); the caller splits on time
    # order, so hand them back oldest-first.
    return list(reversed(examples))


def _forward_split(*groups: list, val_frac: float = 0.2) -> tuple[list, list]:
    """Hold out the most recent `val_frac` of each group, keeping time order.

    A plain tail split put every counterfactual in the validation set, because
    they are appended last — so the previous fix shuffled the whole trainset.
    That balanced the slices but destroyed the temporal separation: a validation
    example could predate a training one, which lets the selected instructions
    be scored partly on setups they were effectively fitted to, and makes the
    winning candidate's val score optimistic against true forward performance.

    Splitting each group at its own boundary gives both properties at once —
    every slice sees both kinds of example, and within each kind validation is
    strictly later than training. Each group must arrive in chronological order.
    """
    train: list = []
    val: list = []
    for group in groups:
        if not group:
            continue
        cut = max(1, int(len(group) * (1 - val_frac)))
        train.extend(group[:cut])
        val.extend(group[cut:])
    if not val:  # tiny trainset — never hand MIPRO an empty validation set
        val = train[-1:]
    return train, val


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
        trainset.append(
            _make_example(inputs, "BUY" if t.pnl_pct > 0 else "PASS", t.rrr_achieved)
        )

    if len(trainset) < MIN_REAL_EXAMPLES:
        logger.info(
            "MIPRO [%s]: only %d real examples, need %d — skipping",
            track, len(trainset), MIN_REAL_EXAMPLES,
        )
        return False

    # Augment with counterfactually-labeled PASS decisions, capped at the number
    # of real-trade examples so hindsight labels can't dominate lived outcomes.
    counterfactuals: list = []
    try:
        counterfactuals = _build_counterfactual_examples(
            track, min(settings.counterfactual_max_examples, len(trainset))
        )
    except Exception as exc:
        logger.warning("MIPRO [%s]: counterfactual build failed (continuing without): %s", track, exc)

    total = len(trainset) + len(counterfactuals)
    if total < MIN_EXAMPLES_FOR_OPTIMIZATION:
        logger.info(
            "MIPRO [%s]: %d examples (%d real + %d counterfactual), need %d — skipping",
            track, total, len(trainset), len(counterfactuals), MIN_EXAMPLES_FOR_OPTIMIZATION,
        )
        return False

    train, val = _forward_split(trainset, counterfactuals)

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
        # Historical book statistics — NOT a score for the program just compiled.
        # This line used to read "optimization metric = ...", which looked like
        # the optimizer's result but is win_rate * avg_rrr over every past trade,
        # computed after the compile and unchanged by it: it would print the same
        # number if MIPRO had produced nonsense. Whether the new program is any
        # good is answered later, by grouping closed trades on program_hash.
        metrics = compute_metrics(portfolio)
        logger.info(
            "MIPRO [%s]: compiled and applied (program %s). Book to date: "
            "win_rate=%.1f%%, avg_rrr=%.2f over %d trades — prior performance, "
            "not a score for this program.",
            track, program_fingerprint(out_path) or BASELINE,
            metrics.win_rate * 100, metrics.avg_rrr, metrics.total_trades,
        )

        # Offsite backup of the new program (best-effort, never fails the run)
        from src.scheduler.backup import backup_compiled_program
        backup_compiled_program(track, metrics.to_dict())

        return True

    except Exception as exc:
        logger.error("MIPRO optimization error for %s track: %s", track, exc, exc_info=True)
        return False


def run_heuristic_refinement(track: str) -> None:
    """Weekly maintenance: score skipped setups, prune, promote and demote."""
    from src.agent.memory import get_store
    store = get_store(track)
    # Score the setups these rules argued against before anything is pruned or
    # promoted on the strength of a score that only counted opened trades.
    try:
        scored = score_heuristics_from_decisions(track)
    except Exception as exc:
        logger.warning("Skipped-setup scoring failed for %s: %s", track, exc)
        scored = 0
    # Dedupe next: a cluster's copies should be retired before pruning decides
    # what has earned its place, so survivors are judged against distinct rules.
    deduped = store.dedupe()
    pruned = store.prune()
    promoted, demoted = store.promote_core()
    logger.info(
        "Heuristic refinement [%s]: scored=%d, deduped=%d, pruned=%d, "
        "promoted_to_core=%d, demoted=%d",
        track, scored, deduped, pruned, promoted, demoted,
    )
