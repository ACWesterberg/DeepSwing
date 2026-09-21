"""
MIPROv2 optimization of the DSPy decision program.

`run_mipro_optimization` trains on lived trades plus counterfactually-labelled
PASS/BLOCKED decisions, so the trainset covers both sides of the decision.
Labelling a skipped setup only needs the underlying's forward path, which
`_label_forward_path` simulates against the stop and target the system would
have used.
"""
from __future__ import annotations

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Literal

import dspy
from dspy.teleprompt import MIPROv2

from config.settings import settings
from src.agent.compiled_program import BASELINE, program_fingerprint, save_compiled_program
from src.agent.decision import TradeDecision, build_lm
from src.agent.evaluation import (
    corpus_fingerprint, evaluate_program, promotion_decision, summarize_actions,
    temporal_split, write_json_atomic,
)
from src.agent.outcomes import evaluate_forward_path, label_forward_path as _label_forward_path
from src.portfolio.metrics import decision_metric
from src.portfolio.simulator import get_portfolio

# The metric lives in metrics.py so the offline replay harness can import it
# without pulling in dspy; MIPRO and the harness must score identically.
_pnl_weighted_metric = decision_metric

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

def counterfactual_cap(real_examples: int) -> int:
    """How many counterfactual examples may join `real_examples` lived ones.

    A multiple rather than parity. Parity tied the trainset to the scarcest
    input — a live run discarded 60 of 90 available labelled PASS decisions and
    left MIPRO selecting instructions on a 12-example validation split. PASS
    decisions accumulate far faster than closed trades and are labelled from
    price data alone, so the ratio lets the trainset grow with decisions while
    keeping lived outcomes materially represented. MIN_REAL_EXAMPLES separately
    stops a trainset that is purely hindsight.
    """
    return min(
        settings.counterfactual_max_examples,
        int(real_examples * settings.counterfactual_ratio_cap),
    )


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


def _date_example(example, *, decision_time: datetime, available_at: datetime,
                  ticker: str, market: str, source: str, example_id: str):
    for key, value in {
        "decision_time": decision_time.isoformat(), "label_available_at": available_at.isoformat(),
        "ticker": ticker, "market": market, "source": source, "example_id": example_id,
    }.items():
        example[key] = value
    return example


def _counterfactual_available_at(decision_time: datetime) -> datetime:
    horizon_end = decision_time + timedelta(days=settings.counterfactual_horizon_days)
    return horizon_end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


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

            outcome = evaluate_forward_path(window, row.price, row.atr, row.market)
            if outcome is None:
                continue
            fwd_return = outcome.pnl_pct

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
                "id": r.id,
                "entry_inputs": r.entry_inputs,
            }
            for r in rows
        ]
    finally:
        session.close()

    # Same frequency-ordered, per-ticker-capped sampling the replay harness
    # uses. Taking the most recent N instead let a handful of frequently
    # decided tickers supply most of the counterfactual half — one blob per
    # (track, ticker) per day means a name decided daily for six weeks
    # contributes ~45 near-identical rows. The composition cap (hindsight vs
    # lived) is a separate concern and stays with counterfactual_ratio_cap.
    from src.agent.replay import select_decision_rows
    decisions = select_decision_rows(
        decisions, limit=max_examples, max_per_ticker=5, max_tickers=150,
    )

    decisions.sort(key=lambda d: d["timestamp"])
    examples: list = []
    ohlcv_cache: dict[str, object] = {}
    for d in decisions:
        if len(examples) >= max_examples:
            break
        ticker = d["ticker"]
        available_at = _counterfactual_available_at(d["timestamp"])
        if available_at > datetime.utcnow():
            continue
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

        labeled = _label_forward_path(window, d["price"], d.get("atr"), d["market"])
        if labeled is None:
            continue
        label, fwd_return = labeled
        examples.append(_date_example(
            _make_example(d["entry_inputs"], label, fwd_return),
            decision_time=d["timestamp"], available_at=available_at,
            ticker=ticker, market=d["market"], source="counterfactual",
            example_id=f"{track}:decision:{d['id']}",
        ))

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

    # Fail before spending anything. MIPROv2 imports optuna deep inside
    # compile(), after bootstrapping demos and proposing instructions with the
    # heavy prompt model — so a missing dependency costs a full proposer run
    # every week and still produces nothing.
    try:
        import optuna  # noqa: F401
    except ImportError:
        logger.error(
            "MIPRO [%s]: optuna is not installed — MIPROv2 cannot run. "
            "Install it (pip install optuna) and the next weekly run will compile.",
            track,
        )
        return False

    logger.info("MIPRO [%s]: starting optimization with %d trades", track, len(trades))

    if sum(bool(getattr(t, "entry_inputs", None)) for t in trades) < MIN_REAL_EXAMPLES:
        logger.info("MIPRO [%s]: insufficient real trades with recorded inputs", track)
        return False
    # All opportunities now use the same horizon/executor, including historical BUYs.
    # Actual managed-trade returns cannot serve as labels for alternative plans.
    try:
        examples = _build_plan_examples(track)
    except Exception as exc:
        logger.error("MIPRO [%s]: plan corpus build failed: %s", track, exc)
        return False
    if len(examples) < MIN_EXAMPLES_FOR_OPTIMIZATION:
        logger.info("MIPRO [%s]: insufficient complete plan paths", track)
        return False
    return _compile_and_evaluate(track, examples)


def _build_plan_examples(track: TrackType) -> list:
    from src.agent.replay import build_corpus
    from src.agent.plan_replay import evaluate_plan, fixed_prediction
    corpus = build_corpus(track=track, plans=True,
                          limit=settings.counterfactual_max_examples + 500)
    real = [e for e in corpus if e.source_action == "BUY"]
    skipped = [e for e in corpus if e.source_action != "BUY"][:counterfactual_cap(len(real))]
    examples = []
    for row in real + skipped:
        outcome = evaluate_plan(row.plan_path, fixed_prediction(row.plan_path))
        decision_time = datetime.fromisoformat(row.timestamp)
        example = _date_example(
            _make_example(row.entry_inputs, "BUY" if outcome["net_r"] > 0 else "PASS", outcome["net_r"]),
            decision_time=decision_time, available_at=_counterfactual_available_at(decision_time),
            ticker=row.ticker, market=row.market, source="real" if row.source_action == "BUY" else "counterfactual",
            example_id=f"{track}:{row.ticker}:{row.timestamp}",
        )
        example["plan_path"] = row.plan_path
        examples.append(example)
    return examples


def _compile_and_evaluate(track: TrackType, examples: list) -> bool:
    now = datetime.utcnow()
    run_id = now.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    evaluation_dir = settings.compiled_dir / "evaluations" / track
    run_dir = evaluation_dir / run_id
    state_path = evaluation_dir / "state.json"
    out_path = settings.compiled_dir / f"{track}_trade_decision.json"
    report = {
        "run_id": run_id, "track": track, "created_at": now.isoformat(), "status": "preparing",
        "scope": "entry selection on stored outcomes; not a test of predicted stop/target payoff",
    }
    try:
        from src.agent.plan_replay import evaluate_plan_examples, plan_metric, validate_path
        plan_mode = any(e.get("plan_path") is not None for e in examples)
        if plan_mode:
            for e in examples:
                validate_path(e["plan_path"])
            report["scope"] = "isolated daily trade plans; fixed ATR risk unit; no portfolio capacity or LLM exits"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state_path.exists() and (not isinstance(state, dict) or state.get("version") != 1 or not state.get("last_test_time")):
            raise ValueError("Invalid evaluation state; refusing to reuse possibly exposed test data")
        split = temporal_split(examples, now=now, last_test_time=state.get("last_test_time"))
        report["split"] = split.summary()
        report["previous_test_watermark"] = state.get("last_test_time")
        requirements = {
            "training_examples": len(split.train) >= settings.mipro_min_train_examples,
            "real_training_examples": sum(e["source"] == "real" for e in split.train) >= MIN_REAL_EXAMPLES,
            "validation_examples": len(split.validation) >= settings.mipro_min_validation_examples,
            "test_examples": len(split.test) >= settings.promotion_min_examples,
            "test_tickers": report["split"]["test_tickers"] >= settings.promotion_min_tickers,
            "test_days": report["split"]["test_days"] >= settings.promotion_min_days,
        }
        report["sample_checks"] = requirements
        report["thresholds"] = {
            key: getattr(settings, key) for key in (
                "mipro_min_train_examples", "mipro_min_validation_examples", "promotion_min_examples",
                "promotion_min_tickers", "promotion_min_days", "promotion_min_buys",
                "promotion_min_metric_gain", "promotion_bootstrap_samples",
            )
        }
        if not all(requirements.values()):
            report["status"] = "skipped"
            report["reason"] = "Insufficient fresh, non-overlapping evidence: " + ", ".join(k for k, v in requirements.items() if not v)
            write_json_atomic(run_dir / "report.json", report)
            logger.info("MIPRO [%s]: %s", track, report["reason"])
            return False

        datasets = {name: [e.toDict() if hasattr(e, "toDict") else dict(e) for e in values]
                    for name, values in (("train", split.train), ("validation", split.validation), ("test", split.test))}
        report["corpus_hash"] = corpus_fingerprint([row for rows in datasets.values() for row in rows])
        report["label_policy"] = {key: getattr(settings, key) for key in (
            "counterfactual_horizon_days", "atr_stop_multiplier", "min_rrr", "commission_pct",
            "fx_commission_pct", "simulated_slippage", "trailing_stop_atr_multiplier", "breakeven_arm_atr_multiplier",
        )}
        write_json_atomic(run_dir / "corpus.json", {"version": 1, "hash": report["corpus_hash"], "datasets": datasets})

        incumbent_hash = program_fingerprint(out_path)
        if out_path.exists() and incumbent_hash is None:
            raise ValueError("Cannot load the incumbent; refusing to substitute a baseline")
        incumbent = dspy.Predict(TradeDecision)
        if out_path.exists():
            incumbent.load(str(out_path))
        report["incumbent_hash"] = incumbent_hash or BASELINE
        incumbent.set_lm(None)
        save_compiled_program(incumbent, run_dir / "incumbent.json",
                              lambda p: dspy.Predict(TradeDecision).load(str(p)))

        if track == "claude":
            model, prompt_model = settings.claude_decision_model, settings.claude_prompt_model
            key, effort = settings.anthropic_api_key, ""
        else:
            model, prompt_model = settings.gpt_decision_model, settings.gpt_prompt_model
            key, effort = settings.openai_api_key, settings.gpt_decision_reasoning_effort
        if not key:
            raise ValueError(f"No API key configured for {track}")
        report["model"] = {"task": model, "proposer": prompt_model, "reasoning_effort": effort}
        task_lm = build_lm(track, model, key, reasoning_effort=effort)
        prompt_lm = build_lm(track, prompt_model, key, max_tokens=4096)
        report["status"] = "compiling"
        write_json_atomic(run_dir / "report.json", report)
        with dspy.context(lm=task_lm):
            metric = plan_metric(split.train + split.validation) if plan_mode else _pnl_weighted_metric
            optimizer = MIPROv2(metric=metric, prompt_model=prompt_lm,
                                task_model=task_lm, auto="light", num_threads=1)
            # Timing/source metadata stays outside the proposer's training examples.
            compiled = optimizer.compile(
                dspy.Predict(TradeDecision),
                trainset=[_make_example(e, e["action"], e["r_multiple"]) for e in split.train],
                valset=[_make_example(e, e["action"], e["r_multiple"]) for e in split.validation],
                requires_permission_to_run=False,
            )
        compiled.set_lm(None)
        candidate_path = run_dir / "candidate.json"
        save_compiled_program(compiled, candidate_path, lambda p: dspy.Predict(TradeDecision).load(str(p)))
        candidate = dspy.Predict(TradeDecision)
        candidate.load(str(candidate_path))
        report["candidate_hash"] = program_fingerprint(candidate_path)

        # Reserve before observing any test outputs, including on failed evaluations.
        write_json_atomic(state_path, {
            "version": 1, "last_test_time": max(e["decision_time"] for e in split.test), "run_id": run_id,
        })
        report["status"] = "evaluating"
        write_json_atomic(run_dir / "report.json", report)
        results = {
            "always_buy": (evaluate_plan_examples(split.test, reference="always_buy") if plan_mode
                           else summarize_actions(split.test, ["BUY"] * len(split.test))),
            "always_pass": (evaluate_plan_examples(split.test, reference="always_pass") if plan_mode
                            else summarize_actions(split.test, ["PASS"] * len(split.test))),
        }
        report["results"] = results
        for name, program in (("incumbent", incumbent), ("baseline", dspy.Predict(TradeDecision)), ("candidate", candidate)):
            program.set_lm(task_lm)
            results[name] = (evaluate_plan_examples(split.test, program) if plan_mode
                             else evaluate_program(split.test, program))
            write_json_atomic(run_dir / "report.json", report)
        gate = promotion_decision(split.test, results, min_gain=settings.promotion_min_metric_gain,
                                  min_buys=settings.promotion_min_buys,
                                  bootstrap_samples=settings.promotion_bootstrap_samples)
        report["gate"] = gate
        if not gate["promote"]:
            report["status"] = "rejected"
            write_json_atomic(run_dir / "report.json", report)
            logger.info("MIPRO [%s]: candidate rejected by paired promotion gate (%s)", track, run_id)
            return False
        if program_fingerprint(out_path) != incumbent_hash:
            raise ValueError("Incumbent changed during evaluation; refusing to replace it")
        candidate.set_lm(None)
        save_compiled_program(candidate, out_path, lambda p: dspy.Predict(TradeDecision).load(str(p)))
        report["status"] = "promoted"
        write_json_atomic(run_dir / "report.json", report)
        from src.agent.decision import DecisionEngine
        try:
            DecisionEngine.for_track(track).reload()
        except Exception as exc:
            report["reload_error"] = str(exc)
            logger.error("MIPRO [%s]: promoted on disk but runtime reload failed: %s", track, exc)
        logger.info("MIPRO [%s]: promoted %s after independent comparison (%s)",
                    track, report["candidate_hash"], run_id)
        from src.scheduler.backup import backup_compiled_program
        try:
            backup_compiled_program(track, {"promotion_run": run_id, "gate": gate})
        except Exception as exc:
            report["backup_error"] = str(exc)
            logger.error("MIPRO [%s]: promoted but backup failed: %s", track, exc)
        write_json_atomic(run_dir / "report.json", report)
        return True
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        write_json_atomic(run_dir / "report.json", report)
        logger.error("MIPRO [%s]: evaluation failed: %s", track, exc, exc_info=True)
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
