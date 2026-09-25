from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from src.agent.replay import DECISION_INPUTS
from src.portfolio.metrics import decision_metric


def utc_time(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime):
        raise ValueError("A decision and label-availability timestamp are required")
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


@dataclass
class TemporalSplit:
    train: list
    validation: list
    test: list
    validation_start: datetime
    test_start: datetime
    purged: int

    def summary(self) -> dict:
        return {
            "train": len(self.train), "validation": len(self.validation), "test": len(self.test),
            "validation_start": self.validation_start.isoformat(),
            "test_start": self.test_start.isoformat(), "purged": self.purged,
            "test_tickers": len({e["ticker"] for e in self.test}),
            "test_days": len({utc_time(e["decision_time"]).date() for e in self.test}),
        }


def temporal_split(
    examples: list, *, now: datetime, last_test_time: str | None = None,
    validation_fraction: float = 0.2, test_fraction: float = 0.2,
) -> TemporalSplit:
    """Use global decision-time boundaries and purge labels unavailable at each boundary."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 - validation_fraction:
        raise ValueError("Invalid temporal split fractions")
    if len(examples) < 3:
        raise ValueError("At least three dated examples are required")
    ordered = sorted(examples, key=lambda e: utc_time(e["decision_time"]))
    for e in ordered:
        if utc_time(e["label_available_at"]) < utc_time(e["decision_time"]):
            raise ValueError("A label cannot be available before its decision")
    validation_start = utc_time(ordered[int(len(ordered) * (1 - validation_fraction - test_fraction))]["decision_time"])
    test_start = utc_time(ordered[int(len(ordered) * (1 - test_fraction))]["decision_time"])
    watermark = utc_time(last_test_time) if last_test_time else datetime.min
    train, validation, test = [], [], []
    for e in ordered:
        decision = utc_time(e["decision_time"])
        available = utc_time(e["label_available_at"])
        if decision < validation_start and available < validation_start:
            train.append(e)
        elif validation_start <= decision < test_start and available < test_start:
            validation.append(e)
        elif decision >= test_start and decision > watermark and available <= utc_time(now):
            test.append(e)
    return TemporalSplit(train, validation, test, validation_start, test_start,
                         len(ordered) - len(train) - len(validation) - len(test))


def summarize_actions(examples: list, actions: list[str]) -> dict:
    if not examples or len(examples) != len(actions):
        raise ValueError("Every example needs exactly one prediction")
    scores, taken = [], []
    for e, action in zip(examples, actions):
        if action not in ("BUY", "PASS") or not math.isfinite(e["r_multiple"]):
            raise ValueError("Invalid prediction or outcome")
        scores.append(decision_metric(SimpleNamespace(r_multiple=e["r_multiple"]), SimpleNamespace(action=action)))
        if action == "BUY":
            taken.append(e["r_multiple"])
    return {
        "n": len(examples), "mean_metric": sum(scores) / len(scores),
        "buys": len(taken), "total_r": sum(taken),
        "mean_r": sum(taken) / len(taken) if taken else 0.0,
        "actions": actions, "scores": scores,
    }


def evaluate_program(examples: list, program: Callable) -> dict:
    """Evaluate complete structured outputs without treating failures as decisions."""
    actions = []
    for e in examples:
        prediction = program(**{key: e[key] for key in DECISION_INPUTS})
        action = str(prediction.action).upper()
        confidence, stop, target = (float(getattr(prediction, key)) for key in ("confidence", "stop_loss", "target"))
        if not all(math.isfinite(v) for v in (confidence, stop, target)) or not 0 <= confidence <= 1:
            raise ValueError(f"Invalid structured prediction for {e['example_id']}")
        if action == "BUY" and not 0 < stop < target:
            raise ValueError(f"Invalid BUY levels for {e['example_id']}")
        actions.append(action)
    return summarize_actions(examples, actions)


def _cluster_lower_bound(differences: np.ndarray, groups: list[str], samples: int) -> float:
    keys = sorted(set(groups))
    sums = np.array([sum(differences[i] for i, g in enumerate(groups) if g == key) for key in keys])
    counts = np.array([groups.count(key) for key in keys])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(keys), size=(samples, len(keys)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.quantile(means, 0.05))


def promotion_decision(
    examples: list, results: dict, *, min_gain: float, min_buys: int, bootstrap_samples: int = 1000,
    required_references: set[str] | None = None,
) -> dict:
    """Require paired improvement over the incumbent, baseline and trivial references."""
    required = required_references or {"candidate", "incumbent", "baseline", "always_buy", "always_pass"}
    if not examples or not required.issubset(results):
        raise ValueError("Promotion requires examples and all reference results")
    for result in results.values():
        if len(result["scores"]) != len(examples) or not all(math.isfinite(s) for s in result["scores"]):
            raise ValueError("Promotion requires complete finite paired scores")
    candidate = results["candidate"]
    ticker_groups = [e["ticker"] for e in examples]
    day_groups = [utc_time(e["decision_time"]).date().isoformat() for e in examples]
    comparisons = {}
    for name, result in results.items():
        if name == "candidate":
            continue
        differences = np.array(candidate["scores"]) - np.array(result["scores"])
        lower = min(_cluster_lower_bound(differences, ticker_groups, bootstrap_samples),
                    _cluster_lower_bound(differences, day_groups, bootstrap_samples))
        comparisons[name] = {"gain": float(differences.mean()), "lower_bound": lower}
    enough_buys = candidate["buys"] >= min_buys
    positive_r = candidate["total_r"] > 0
    beats_references = all(c["gain"] >= min_gain and c["lower_bound"] > 0 for c in comparisons.values())
    return {
        "promote": enough_buys and positive_r and beats_references,
        "enough_buys": enough_buys, "positive_total_r": positive_r,
        "comparisons": comparisons,
        "uncertainty_method": "paired ticker/day cluster bootstrap; conservative minimum of one-sided 95% bounds",
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def corpus_fingerprint(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
