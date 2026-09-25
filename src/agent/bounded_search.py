"""Bounded single-instruction search with durable exact-request caching."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from config.settings import settings
from src.agent.evaluation import write_json_atomic
from src.agent.provider_usage import usage_from_lm_history
from src.agent.single_request import single_predict
from src.agent.replay import DECISION_INPUTS


def proposal_fields(training, instruction):
    return {
        "training_summary": compact_training_summary(training),
        "existing_instruction": instruction,
        "mandate": (
            "Long-only swing entries. Preserve BUY/PASS outputs and honest structural stop/target placement. "
            "Do not use future outcomes, change risk limits, or claim profitability. Return one instruction."
        ),
    }


def compact_training_summary(examples: list) -> str:
    """Aggregate training evidence without exposing future paths or the universe."""
    actions = Counter(str(e["action"]) for e in examples)
    sources = Counter(str(e["source"]) for e in examples)
    markets = Counter(str(e["market"]) for e in examples)
    outcomes = [float(e["r_multiple"]) for e in examples]
    payload = {
        "examples": len(examples), "actions": dict(actions), "sources": dict(sources),
        "markets": dict(markets),
        "outcome_r": {
            "mean": sum(outcomes) / len(outcomes), "positive": sum(v > 0 for v in outcomes),
            "zero_or_negative": sum(v <= 0 for v in outcomes),
            "min": min(outcomes), "max": max(outcomes),
        },
    }
    return json.dumps(payload, sort_keys=True, allow_nan=False)


def select_screen_examples(examples: list, count: int) -> list:
    """Choose a chronological spread without optimizing selection on outcomes."""
    if count >= len(examples):
        return list(examples)
    indexes = [round(i * (len(examples) - 1) / (count - 1)) for i in range(count)]
    return [examples[i] for i in indexes]


class SearchBudget:
    def __init__(self, path: Path, *, max_requests: int, max_input_bytes: int,
                 max_reserved_output_tokens: int):
        self.path = path
        self.max_requests = max_requests
        self.max_input_bytes = max_input_bytes
        self.max_reserved_output_tokens = max_reserved_output_tokens
        self.state = (json.loads(path.read_text()) if path.exists() else
                      {"version": 1, "requests": 0, "input_bytes": 0, "reserved_output_tokens": 0})
        if self.state.get("version") != 1 or any(
            type(self.state.get(k)) is not int or self.state[k] < 0
            for k in ("requests", "input_bytes", "reserved_output_tokens")
        ):
            raise ValueError("Invalid durable search budget")
        write_json_atomic(path, self.state)

    def reserve(self, input_bytes: int, output_tokens: int) -> None:
        proposed = {
            "requests": self.state["requests"] + 1,
            "input_bytes": self.state["input_bytes"] + input_bytes,
            "reserved_output_tokens": self.state["reserved_output_tokens"] + output_tokens,
        }
        if proposed["requests"] > self.max_requests:
            raise RuntimeError("Bounded-search request limit reached")
        if proposed["input_bytes"] > self.max_input_bytes:
            raise RuntimeError("Bounded-search input-byte limit reached")
        if proposed["reserved_output_tokens"] > self.max_reserved_output_tokens:
            raise RuntimeError("Bounded-search output-token reservation limit reached")
        self.state.update(proposed)
        write_json_atomic(self.path, self.state)


def exact_request(
    track: str,
    payload: dict,
    *,
    output_tokens: int,
    budget: SearchBudget,
    call: Callable[[], dict],
    lm=None,
) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    request_id = hashlib.sha256(canonical.encode()).hexdigest()
    path = settings.compiled_dir / "search_cache" / track / f"{request_id}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("status") == "complete":
            return record["response"]
        if record.get("status") != "retry_authorized":
            raise RuntimeError(f"Cached bounded-search request is {record.get('status', 'invalid')}; explicit retry required")
        previous = {
            "status": record.get("previous_status", "failed"), "error": record.get("error"),
            "retry_authorization": record.get("retry_authorization"),
        }
        attempts = list(record.get("attempts", []))
    else:
        previous = None
        attempts = []
    budget.reserve(len(canonical.encode()), output_tokens)
    record = {
        "version": 1, "request_id": request_id, "status": "started",
        "payload": payload, "reserved_output_tokens": output_tokens,
        "started_at": datetime.utcnow().isoformat(), "actual_usage": None,
        "attempts": attempts,
    }
    if previous:
        record["previous_attempt"] = previous
    attempt = {"status": "started", "started_at": record["started_at"],
               "reserved_output_tokens": output_tokens, "actual_usage": None}
    record["attempts"].append(attempt)
    write_json_atomic(path, record)
    history_start = len(getattr(lm, "history", [])) if lm is not None else 0
    try:
        response = call()
        json.dumps(response, allow_nan=False)
        record.update(status="complete", response=response, completed_at=datetime.utcnow().isoformat())
    except Exception as exc:
        record.update(status="failed", error=str(exc), failed_at=datetime.utcnow().isoformat())
        raise
    finally:
        record["actual_usage"] = usage_from_lm_history(lm, history_start) if lm is not None else None
        attempt.update({
            "status": record["status"], "started_at": record["started_at"],
            "finished_at": record.get("completed_at") or record.get("failed_at"),
            "reserved_output_tokens": output_tokens, "actual_usage": record["actual_usage"],
        })
        write_json_atomic(path, record)
    return response


def authorize_search_retry(track: str, request_id: str, *, authorized_by: str, reason: str) -> Path:
    """Authorize one retry of an exact failed request; started requests stay ambiguous."""
    if not authorized_by.strip() or not reason.strip():
        raise ValueError("Retry authorization requires an approver and reason")
    path = settings.compiled_dir / "search_cache" / track / f"{request_id}.json"
    if not path.exists():
        raise ValueError("Bounded-search request does not exist")
    record = json.loads(path.read_text())
    if record.get("status") != "failed":
        raise ValueError("Only failed bounded-search requests can be retried")
    record["previous_status"] = record["status"]
    record["status"] = "retry_authorized"
    record["retry_authorization"] = {
        "authorized_at": datetime.utcnow().isoformat(),
        "authorized_by": authorized_by.strip(), "reason": reason.strip(),
    }
    write_json_atomic(path, record)
    return path


def cached_program(
    track: str, program, program_hash: str, model: str, budget: SearchBudget,
    output_tokens: int, lm=None,
):
    def predict(**inputs):
        request = {
            "kind": "evaluation", "track": track, "model": model,
            "program_hash": program_hash,
            "inputs": {key: inputs[key] for key in DECISION_INPUTS},
            "output_tokens": output_tokens,
        }

        def invoke():
            result = single_predict(program, lm if lm is not None else program.get_lm(), inputs)
            return {
                "action": str(result.action), "confidence": float(result.confidence),
                "stop_loss": float(result.stop_loss), "target": float(result.target),
                "reasoning": str(getattr(result, "reasoning", "")),
            }

        return SimpleNamespace(**exact_request(
            track, request, output_tokens=output_tokens, budget=budget, call=invoke,
            lm=lm if lm is not None else program.get_lm(),
        ))
    return predict
