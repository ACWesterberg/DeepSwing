from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from config.settings import BASE_DIR, settings
from src.backtesting.replay import DEFAULT_TRAINSET_PATH

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]

BACKTEST_COMPILED_DIR = BASE_DIR / "compiled" / "backtest"

MIN_EXAMPLES = 30

# Keep in sync with src/scheduler/optimizer.py — duplicated so the backtest
# never imports the live portfolio/DB chain.
_PNL_METRIC_SCALE = 10.0


def pnl_weighted_metric(example, prediction, trace=None) -> float:
    """P&L-weighted decision reward, mirroring the live MIPRO metric."""
    pred_action = str(getattr(prediction, "action", "")).upper()
    pnl = float(getattr(example, "pnl_pct", 0.0) or 0.0)
    realized = pnl if pred_action == "BUY" else 0.0
    return 0.5 + 0.5 * math.tanh(realized * _PNL_METRIC_SCALE)


def load_records(path: Path) -> list[dict]:
    """Parse a replay trainset JSONL into records, dropping malformed lines."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                r["pnl_pct"] = float(r["pnl_pct"])
                if not isinstance(r["inputs"], dict):
                    raise KeyError("inputs")
                records.append(r)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed trainset line %d: %s", n, exc)
    records.sort(key=lambda r: r.get("scan_date", ""))
    return records


def dataset_stats(records: list[dict]) -> dict:
    if not records:
        return {"examples": 0}
    winners = sum(1 for r in records if r["pnl_pct"] > 0)
    by_market: dict[str, int] = {}
    by_exit: dict[str, int] = {}
    for r in records:
        by_market[r.get("market", "?")] = by_market.get(r.get("market", "?"), 0) + 1
        by_exit[r.get("exit_reason", "?")] = by_exit.get(r.get("exit_reason", "?"), 0) + 1
    return {
        "examples": len(records),
        "winners": winners,
        "losers": len(records) - winners,
        "avg_pnl_pct": round(sum(r["pnl_pct"] for r in records) / len(records) * 100, 3),
        "date_range": [records[0].get("scan_date"), records[-1].get("scan_date")],
        "by_market": by_market,
        "by_exit_reason": by_exit,
    }


def records_to_examples(records: list[dict]) -> list:
    import dspy

    examples = []
    for r in records:
        inputs = r["inputs"]
        examples.append(
            dspy.Example(
                technicals=inputs.get("technicals", ""),
                regime=inputs.get("regime", ""),
                news_summary=inputs.get("news_summary", ""),
                macro_context=inputs.get("macro_context", ""),
                heuristics=inputs.get("heuristics", ""),
                action="BUY" if r["pnl_pct"] > 0 else "PASS",
                pnl_pct=r["pnl_pct"],
            ).with_inputs("technicals", "regime", "news_summary", "macro_context", "heuristics")
        )
    return examples


def run_backtest_mipro(
    track: TrackType,
    dataset_path: Path = DEFAULT_TRAINSET_PATH,
    num_threads: int = 1,
) -> Optional[Path]:
    """MIPROv2 over a replay-generated trainset. Saves to compiled/backtest/ —
    never touches the live compiled program or the live DecisionEngine."""
    import dspy
    from dspy.teleprompt import MIPROv2

    from src.agent.decision import TradeDecision, build_lm

    records = load_records(dataset_path)
    if len(records) < MIN_EXAMPLES:
        logger.error(
            "Backtest MIPRO [%s]: only %d examples in %s, need %d",
            track, len(records), dataset_path, MIN_EXAMPLES,
        )
        return None

    stats = dataset_stats(records)
    logger.info("Backtest MIPRO [%s]: %s", track, stats)

    trainset = records_to_examples(records)
    # Chronological split — validate on the most recent period, no time leakage
    split = int(len(trainset) * 0.8)
    train, val = trainset[:split], trainset[split:]

    if track == "claude":
        task_lm = build_lm(track, settings.claude_decision_model, settings.anthropic_api_key)
        prompt_lm = build_lm(track, settings.claude_prompt_model, settings.anthropic_api_key, max_tokens=4096)
    else:
        task_lm = build_lm(track, settings.gpt_decision_model, settings.openai_api_key)
        prompt_lm = build_lm(track, settings.gpt_prompt_model, settings.openai_api_key, max_tokens=4096)

    program = dspy.Predict(TradeDecision)

    dspy.configure(lm=task_lm)
    optimizer = MIPROv2(
        metric=pnl_weighted_metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto="light",
        num_threads=num_threads,
    )
    compiled = optimizer.compile(
        program,
        trainset=train,
        valset=val,
        requires_permission_to_run=False,
    )

    out_path = BACKTEST_COMPILED_DIR / f"{track}_trade_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path.rename(BACKTEST_COMPILED_DIR / f"{track}_trade_decision_{stamp}.json")

    compiled.save(str(out_path))
    meta = {
        "track": track,
        "dataset": str(dataset_path),
        "dataset_stats": stats,
        "train_size": len(train),
        "val_size": len(val),
        "compiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Backtest MIPRO [%s]: saved %s", track, out_path)
    logger.info(
        "To adopt for live trading: cp %s %s",
        out_path, settings.compiled_dir / f"{track}_trade_decision.json",
    )
    return out_path
