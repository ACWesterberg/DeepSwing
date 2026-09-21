"""Frozen daily-bar trade-plan experiments. Scores are not portfolio returns."""
from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from config.settings import settings
from src.portfolio.daily import DailyPortfolio
from src.portfolio.simulator import commission_per_leg

PLAN_VERSION = 1


@lru_cache(maxsize=1)
def execution_code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in ("agent/plan_replay.py", "portfolio/daily.py", "portfolio/simulator.py"):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PlanPolicy:
    commission: float
    slippage: float
    atr_stop: float
    min_rrr: float
    min_stop_atr: float
    min_stop_cost: float
    trailing_atr: float
    breakeven_atr: float

    @classmethod
    def current(cls, market: str):
        return cls(commission_per_leg(market), settings.simulated_slippage,
                   settings.atr_stop_multiplier, settings.min_rrr,
                   settings.min_stop_atr_multiplier, settings.min_stop_cost_multiple,
                   settings.trailing_stop_atr_multiplier, settings.breakeven_arm_atr_multiplier)

    def validate(self):
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Invalid frozen execution policy")
        if self.commission >= 1 or self.slippage >= 1 or self.atr_stop <= 0 or self.min_rrr <= 0:
            raise ValueError("Invalid frozen execution policy")


def freeze_path(window, price: float, atr: float, market: str, decision_time: datetime) -> dict:
    """Require complete OHLC bars strictly after the decision's calendar day."""
    payload = {
        "version": PLAN_VERSION, "price": price, "atr": atr,
        "execution_code_hash": execution_code_hash(),
        "decision_time": decision_time.isoformat(),
        "policy": asdict(PlanPolicy.current(market)),
        "bars": [{"date": stamp.date().isoformat(), **{key.lower(): float(row[key])
                 for key in ("Open", "High", "Low", "Close")}}
                 for stamp, row in window.sort_index().iterrows()],
    }
    validate_path(payload)
    return payload


def validate_path(path: dict) -> PlanPolicy:
    if path.get("version") != PLAN_VERSION:
        raise ValueError("Unsupported trade-plan path version; rebuild the corpus")
    if path.get("execution_code_hash") != execution_code_hash():
        raise ValueError("Trade-plan execution code changed; rebuild the corpus or use its original code revision")
    policy = PlanPolicy(**path["policy"])
    policy.validate()
    if not all(math.isfinite(path[k]) and path[k] > 0 for k in ("price", "atr")):
        raise ValueError("Entry price and ATR must be finite and positive")
    previous = datetime.fromisoformat(path["decision_time"]).date()
    if not path["bars"]:
        raise ValueError("Trade-plan evaluation requires a forward OHLC path")
    for bar in path["bars"]:
        day = datetime.fromisoformat(bar["date"]).date()
        if day <= previous:
            raise ValueError("Forward bars must be unique and strictly after the decision day")
        previous = day
        o, h, l, c = (bar[k] for k in ("open", "high", "low", "close"))
        if not all(math.isfinite(v) and v > 0 for v in (o, h, l, c)) or not l <= min(o, c) <= max(o, c) <= h:
            raise ValueError("Invalid OHLC geometry")
    return policy


def fixed_prediction(path: dict, action: str = "BUY"):
    policy = validate_path(path)
    risk = policy.atr_stop * path["atr"]
    return SimpleNamespace(action=action, confidence=1.0,
                           stop_loss=path["price"] - risk, target=path["price"] + policy.min_rrr * risk)


def check_plan(path: dict, prediction) -> dict:
    """Validate a proposal without consulting its forward payoff."""
    policy = validate_path(path)
    action = str(prediction.action).upper()
    confidence, stop, target = (float(getattr(prediction, k)) for k in ("confidence", "stop_loss", "target"))
    if action not in ("BUY", "PASS") or not all(math.isfinite(v) for v in (confidence, stop, target)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid structured trade-plan prediction")
    result = dict(action=action, confidence=confidence, stop_loss=stop, target=target,
                  executed=False, reason="pass", net_r=0.0, pnl_pct=0.0, score=0.5)
    if action == "PASS":
        return result
    price, atr = path["price"], path["atr"]
    # Static live risk rules, evaluated using the corpus's frozen settings.
    cost = (1 + policy.slippage) * (1 + policy.commission) - (1 - policy.slippage) * (1 - policy.commission)
    distance = price - stop
    reason = None
    if not 0 < stop < price < target:
        reason = "invalid_levels"
    elif distance / price < max(policy.min_stop_cost * cost, policy.min_stop_atr * atr / price):
        reason = "stop_inside_floor"
    elif distance > policy.atr_stop * atr * 1.10:
        reason = "stop_beyond_ceiling"
    elif (target - price) / distance < policy.min_rrr:
        reason = "rrr_below_minimum"
    if reason:
        # A blocked BUY earns no profit and is visible separately from a PASS.
        return {**result, "reason": reason}
    return {**result, "reason": "eligible"}


def evaluate_plan(path: dict, prediction) -> dict:
    result = check_plan(path, prediction)
    if result["reason"] != "eligible":
        return result
    policy = PlanPolicy(**path["policy"])
    price, atr = path["price"], path["atr"]
    stop, target = result["stop_loss"], result["target"]
    portfolio = DailyPortfolio(price * 2, policy.commission, policy.slippage,
                               trailing_atr_multiplier=policy.trailing_atr,
                               breakeven_arm_atr_multiplier=policy.breakeven_atr)
    portfolio.open_position("plan", price, stop, target, 1,
                            datetime.fromisoformat(path["decision_time"]).date(),
                            trail_distance=policy.trailing_atr * atr)
    if not portfolio.has_ticker("plan"):
        raise ValueError("Frozen policy cannot fund the one-share experiment")
    for bar in path["bars"]:
        portfolio.update({"plan": bar}, datetime.fromisoformat(bar["date"]).date())
        if portfolio.closed_trades:
            break
    if not portfolio.closed_trades:
        last = path["bars"][-1]
        portfolio.close_position("plan", last["close"], datetime.fromisoformat(last["date"]).date(), "horizon")
    trade = portfolio.closed_trades[0]
    # A fixed, pre-decision risk denominator prevents tighter-stop score inflation.
    net_r = trade.net_pnl / (policy.atr_stop * atr)
    if abs(net_r) < 1e-10:
        net_r = 0.0
    return {**result, "executed": True, "reason": trade.exit_reason,
            "net_r": net_r, "pnl_pct": trade.pnl_pct,
            "score": 0.5 + math.atan(net_r) / math.pi}


def summarize_plans(outcomes: list[dict]) -> dict:
    if not outcomes:
        raise ValueError("Trade-plan scoring requires a nonempty corpus")
    taken = [o for o in outcomes if o["executed"]]
    total = sum(o["net_r"] for o in outcomes)
    return dict(n=len(outcomes), buys=len(taken), proposed_buys=sum(o["action"] == "BUY" for o in outcomes),
                blocked_buys=sum(o["action"] == "BUY" and not o["executed"] for o in outcomes),
                total_r=total, mean_r=total / len(taken) if taken else 0.0,
                mean_metric=sum(o["score"] for o in outcomes) / len(outcomes),
                actions=[o["action"] for o in outcomes], scores=[o["score"] for o in outcomes], outcomes=outcomes)


def score_plans(examples: list, program=None, *, reference: str | None = None) -> dict:
    from src.agent.replay import DECISION_INPUTS
    if reference not in (None, "always_buy", "always_pass"):
        raise ValueError("No plan oracle: fixed-plan labels are not an upper bound for alternative plans")
    # Fail before paid inference if any path is missing or malformed.
    for example in examples:
        if not example.plan_path:
            raise ValueError("Corpus has no frozen trade-plan paths; rebuild with --plans")
        validate_path(example.plan_path)
    outcomes = []
    for example in examples:
        prediction = (fixed_prediction(example.plan_path, "BUY" if reference == "always_buy" else "PASS")
                      if reference else program(**{key: example.entry_inputs[key] for key in DECISION_INPUTS}))
        outcomes.append(evaluate_plan(example.plan_path, prediction))
    return summarize_plans(outcomes)


def evaluate_plan_examples(examples: list, program=None, *, reference: str | None = None) -> dict:
    """Adapter for dated optimizer examples; never send paths to the model."""
    from src.agent.replay import DECISION_INPUTS
    rows = [SimpleNamespace(plan_path=e.get("plan_path"),
                            entry_inputs={key: e[key] for key in DECISION_INPUTS}) for e in examples]
    return score_plans(rows, program, reference=reference)


def plan_metric(examples: list):
    """Keep price paths outside DSPy's examples/demos and instruction proposals."""
    import json
    from src.agent.replay import DECISION_INPUTS

    def key(example):
        return json.dumps({k: example[k] for k in DECISION_INPUTS}, sort_keys=True)

    paths = {}
    for example in examples:
        validate_path(example["plan_path"])
        paths.setdefault(key(example), []).append(example["plan_path"])

    def metric(example, prediction, trace=None):
        try:
            scores = [evaluate_plan(path, prediction)["score"] for path in paths[key(example)]]
            return sum(scores) / len(scores)
        except (ValueError, TypeError, AttributeError, KeyError):
            # Malformed proposals cannot earn credit during search. Held-out
            # scoring is stricter: a malformed prediction aborts the run.
            return 0.0
    return metric
