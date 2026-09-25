"""Prospective, non-executing prompt-candidate shadow evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from config.settings import settings
from src.agent.candidates import candidate_dir
from src.agent.compiled_program import program_fingerprint
from src.agent.evaluation import write_json_atomic
from src.agent.replay import DECISION_INPUTS
from src.agent.single_request import single_predict, effective_output_tokens


def _shadow_root(track: str) -> Path:
    return settings.compiled_dir / "shadow" / track


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _request_id(track: str, candidate_hash: str, ticker: str, decision_time: str, inputs: dict) -> str:
    payload = [track, candidate_hash, ticker, decision_time, {key: inputs[key] for key in DECISION_INPUTS}]
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def pending_candidates(track: str) -> list[dict]:
    root = settings.compiled_dir / "candidates" / track
    found = []
    for path in root.glob("*/manifest.json") if root.exists() else []:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if manifest.get("status") == "pending_forward_evaluation":
            found.append(manifest)
    return sorted(found, key=lambda row: (row.get("registered_at", ""), row["candidate_hash"]))


def _load_ledger(track: str) -> dict:
    path = _shadow_root(track) / "ledger.json"
    if not path.exists():
        return {"version": 1, "days": {}}
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("days"), dict):
        raise ValueError("Invalid shadow request ledger")
    return data


def _reserve(track: str, request_id: str, now: datetime) -> None:
    ledger = _load_ledger(track)
    day = now.date().isoformat()
    usage = ledger["days"].setdefault(day, {"attempts": [], "reserved_output_tokens": 0})
    if request_id in usage["attempts"]:
        raise ValueError("Shadow request attempt was already reserved")
    if len(usage["attempts"]) >= settings.shadow_max_requests_per_day:
        raise RuntimeError("Shadow daily request limit reached")
    reserved = usage["reserved_output_tokens"] + effective_output_tokens(track, settings.shadow_output_tokens_per_request)
    if reserved > settings.shadow_max_reserved_output_tokens_per_day:
        raise RuntimeError("Shadow daily output-token reservation limit reached")
    usage["attempts"].append(request_id)
    usage["reserved_output_tokens"] = reserved
    write_json_atomic(_shadow_root(track) / "ledger.json", ledger)


def _prediction_dict(prediction) -> dict:
    action = str(prediction.action).upper()
    confidence, stop, target = (float(getattr(prediction, key)) for key in ("confidence", "stop_loss", "target"))
    if action not in ("BUY", "PASS"):
        raise ValueError("Invalid shadow action")
    if not all(math.isfinite(v) for v in (confidence, stop, target)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid shadow structured output")
    if action == "BUY" and not 0 < stop < target:
        raise ValueError("Invalid shadow BUY levels")
    return {
        "action": action, "confidence": confidence, "stop_loss": stop, "target": target,
        "reasoning": str(getattr(prediction, "reasoning", "")),
    }


def authorize_shadow_retry(track: str, request_id: str, *, authorized_by: str, reason: str) -> Path:
    """Authorize exactly one new attempt for a retained failed shadow request."""
    if not authorized_by.strip() or not reason.strip():
        raise ValueError("Retry authorization requires an approver and reason")
    path = _shadow_root(track) / "cases" / f"{request_id}.json"
    if not path.exists():
        raise ValueError("Shadow request does not exist")
    case = json.loads(path.read_text())
    if case.get("status") != "failed":
        raise ValueError("Only failed shadow requests can be retried")
    case["status"] = "retry_authorized"
    case["retry_authorization"] = {
        "authorized_at": datetime.utcnow().isoformat(),
        "authorized_by": authorized_by.strip(),
        "reason": reason.strip(),
    }
    write_json_atomic(path, case)
    return path


def _run_candidate(track: str, candidate_hash: str, inputs: dict):
    import dspy

    from src.agent.decision import TradeDecision, build_lm

    program_path = candidate_dir(track, candidate_hash) / "program.json"
    if program_fingerprint(program_path) != candidate_hash:
        raise ValueError("Candidate artifact fingerprint changed")
    program = dspy.Predict(TradeDecision)
    program.load(str(program_path))
    if track == "claude":
        model, key, effort = settings.claude_decision_model, settings.anthropic_api_key, ""
    else:
        model, key, effort = settings.gpt_decision_model, settings.openai_api_key, settings.gpt_decision_reasoning_effort
    if not key:
        raise ValueError(f"No API key configured for {track} shadow evaluation")
    lm = build_lm(track, model, key, max_tokens=settings.shadow_output_tokens_per_request,
                  reasoning_effort=effort)
    program.set_lm(lm)
    return single_predict(program, lm, {key: inputs[key] for key in DECISION_INPUTS})


def collect_shadow_decision(
    *,
    track: str,
    ticker: str,
    market: str,
    decision_time: datetime,
    price: float,
    atr: float,
    entry_inputs: dict,
    incumbent: dict,
    predict: Callable[[str, str, dict], object] | None = None,
) -> dict | None:
    """Evaluate one inactive candidate on the incumbent's exact live inputs.

    The returned candidate output is evidence only. This function never calls
    portfolio or execution code and the scan loop ignores the output.
    """
    if not settings.shadow_enabled:
        return None
    candidates = pending_candidates(track)[: settings.shadow_max_candidates_per_track]
    if not candidates:
        return None
    candidate_hash = candidates[0]["candidate_hash"]
    inputs = {key: entry_inputs[key] for key in DECISION_INPUTS}
    input_bytes = len(_canonical(inputs).encode())
    if input_bytes > settings.shadow_max_input_bytes:
        raise RuntimeError("Shadow input-byte limit exceeded")
    stamp = decision_time.isoformat()
    request_id = _request_id(track, candidate_hash, ticker, stamp, inputs)
    path = _shadow_root(track) / "cases" / f"{request_id}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("status") != "retry_authorized":
            return existing
        attempt = int(existing.get("request", {}).get("attempt", 1)) + 1
    else:
        existing = None
        attempt = 1
    now = datetime.utcnow()
    from src.agent.session_coverage import freeze_sessions
    coverage = freeze_sessions(ticker, market, decision_time.date(),
                               (decision_time + timedelta(days=settings.counterfactual_horizon_days)).date())
    _reserve(track, f"{request_id}:{attempt}", now)
    from src.agent.plan_replay import PlanPolicy, execution_code_hash
    frozen_policy = {
        "execution_code_hash": execution_code_hash(),
        "policy": vars(PlanPolicy.current(market)),
    }
    case = {
        "version": 1, "request_id": request_id, "status": "started",
        "track": track, "candidate_hash": candidate_hash,
        "incumbent_hash": incumbent.get("program_hash", "baseline"),
        "ticker": ticker, "market": market, "decision_time": stamp,
        "horizon_days": settings.counterfactual_horizon_days,
        "label_available_at": (decision_time + timedelta(days=settings.counterfactual_horizon_days + 1)).isoformat(),
        "price": float(price), "atr": float(atr), "entry_inputs": inputs,
        "frozen_execution": frozen_policy,
        "session_coverage": coverage,
        "incumbent": {key: incumbent.get(key) for key in ("action", "confidence", "stop_loss", "target", "reasoning")},
        "request": {
            "reserved_at": now.isoformat(), "input_bytes": input_bytes,
            "reserved_output_tokens": effective_output_tokens(track, settings.shadow_output_tokens_per_request),
            "attempt": attempt, "actual_usage": "unavailable",
        },
    }
    if existing:
        case["previous_attempts"] = existing.get("previous_attempts", []) + [{
            "attempt": existing.get("request", {}).get("attempt", 1),
            "status": "failed", "error": existing.get("error"),
            "retry_authorization": existing.get("retry_authorization"),
        }]
    write_json_atomic(path, case)
    try:
        raw = (predict or _run_candidate)(track, candidate_hash, inputs)
        case["candidate"] = _prediction_dict(raw)
        case["status"] = "awaiting_outcome"
        case["completed_at"] = datetime.utcnow().isoformat()
    except Exception as exc:
        case["status"] = "failed"
        case["error"] = str(exc)
        case["failed_at"] = datetime.utcnow().isoformat()
    write_json_atomic(path, case)
    return case


def collect_mature_shadow_outcomes(track: str, *, now: datetime | None = None) -> dict:
    """Freeze matured paths and score both paired plans without LLM calls."""
    from src.agent.plan_replay import evaluate_plan, validate_path
    from src.data.market_data import fetch_ohlcv

    now = now or datetime.utcnow()
    summary = {"completed": 0, "pending": 0, "failed": 0}
    for path in sorted((_shadow_root(track) / "cases").glob("*.json")):
        case = json.loads(path.read_text())
        if case.get("status") == "failed":
            summary["failed"] += 1
            continue
        if case.get("status") == "complete":
            summary["completed"] += 1
            continue
        if case.get("status") != "awaiting_outcome" or now < datetime.fromisoformat(case["label_available_at"]):
            summary["pending"] += 1
            continue
        try:
            decision_time = datetime.fromisoformat(case["decision_time"])
            end = decision_time + timedelta(days=case["horizon_days"])
            frame = fetch_ohlcv(case["ticker"], case["market"], period="6mo")
            window = frame[(frame.index.date > decision_time.date()) & (frame.index.date <= end.date())]
            from src.agent.session_coverage import freeze_sessions, validate_sessions
            if "session_coverage" not in case:
                case["session_coverage"] = freeze_sessions(case["ticker"], case["market"], decision_time.date(), end.date())
                case["session_coverage_backfilled"] = True
            validate_sessions(case["session_coverage"], window.index.date)
            frozen = {
                "version": 1,
                "price": case["price"], "atr": case["atr"],
                "execution_code_hash": case["frozen_execution"]["execution_code_hash"],
                "decision_time": case["decision_time"],
                "policy": case["frozen_execution"]["policy"],
                "session_coverage": case["session_coverage"],
                "bars": [
                    {"date": stamp.date().isoformat(), **{
                        key.lower(): float(row[key]) for key in ("Open", "High", "Low", "Close")
                    }}
                    for stamp, row in window.sort_index().iterrows()
                ],
            }
            validate_path(frozen)
            outcomes = {}
            for arm in ("incumbent", "candidate"):
                prediction = type("Prediction", (), case[arm])()
                outcomes[arm] = evaluate_plan(frozen, prediction)
            case["path"] = frozen
            case["outcomes"] = outcomes
            case["status"] = "complete"
            case["outcome_collected_at"] = now.isoformat()
            write_json_atomic(path, case)
            summary["completed"] += 1
        except Exception as exc:
            case["outcome_collection_error"] = str(exc)
            case["last_outcome_attempt_at"] = now.isoformat()
            write_json_atomic(path, case)
            summary["pending"] += 1
    return summary


def build_forward_evidence(track: str, candidate_hash: str, *, now: datetime | None = None) -> dict:
    """Aggregate complete paired cases into non-overlapping temporal periods."""
    now = now or datetime.utcnow()
    complete = []
    failed = 0
    ambiguous = 0
    for path in sorted((_shadow_root(track) / "cases").glob("*.json")):
        case = json.loads(path.read_text())
        if case.get("candidate_hash") != candidate_hash:
            continue
        if case.get("status") == "complete":
            complete.append(case)
        elif case.get("status") == "failed":
            failed += 1
        elif case.get("status") == "started":
            ambiguous += 1

    intervals = []
    for case in sorted(complete, key=lambda row: row["decision_time"]):
        start = datetime.fromisoformat(case["decision_time"])
        end = start + timedelta(days=case["horizon_days"])
        if not intervals or start > intervals[-1]["end"]:
            intervals.append({"start": start, "end": end, "cases": [case]})
        elif start == intervals[-1]["start"]:
            intervals[-1]["cases"].append(case)

    periods = []
    all_deltas = []
    candidate_r = incumbent_r = 0.0
    for interval in intervals:
        deltas = [row["outcomes"]["candidate"]["score"] - row["outcomes"]["incumbent"]["score"]
                  for row in interval["cases"]]
        period_candidate_r = sum(row["outcomes"]["candidate"]["net_r"] for row in interval["cases"])
        period_incumbent_r = sum(row["outcomes"]["incumbent"]["net_r"] for row in interval["cases"])
        all_deltas.extend(deltas)
        candidate_r += period_candidate_r
        incumbent_r += period_incumbent_r
        periods.append({
            "start": interval["start"].isoformat(), "end": interval["end"].isoformat(),
            "cases": len(interval["cases"]), "mean_score_delta": sum(deltas) / len(deltas),
            "candidate_total_r": period_candidate_r, "incumbent_total_r": period_incumbent_r,
        })
    mean_gain = sum(p["mean_score_delta"] for p in periods) / len(periods) if periods else 0.0
    checks = {
        "completed_cases": len(complete) >= settings.shadow_min_completed_cases,
        "non_overlapping_periods": len(periods) >= settings.shadow_min_non_overlapping_periods,
        "mean_score_gain": mean_gain >= settings.shadow_min_mean_score_gain,
        "candidate_total_r": candidate_r > 0,
        "no_ambiguous_requests": ambiguous == 0,
        "no_failed_requests": failed == 0,
    }
    report = {
        "version": 1, "generated_at": now.isoformat(), "track": track,
        "candidate_hash": candidate_hash,
        "horizon_days": sorted({case["horizon_days"] for case in complete}),
        "completed_cases": len(complete), "failed_requests": failed,
        "scored_cases": len(all_deltas), "overlapping_cases_excluded": len(complete) - len(all_deltas),
        "ambiguous_requests": ambiguous, "non_overlapping_periods": len(periods),
        "mean_score_gain": mean_gain, "candidate_total_r": candidate_r,
        "incumbent_total_r": incumbent_r, "checks": checks,
        "eligible_for_review": all(checks.values()), "periods": periods,
        "scope": "prospective paired isolated daily trade plans; candidate decisions did not execute",
    }
    write_json_atomic(candidate_dir(track, candidate_hash) / "forward_evidence.json", report)
    return report
