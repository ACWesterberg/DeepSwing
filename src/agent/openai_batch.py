"""Durable OpenAI Batch transport; no uncertain submission is auto-retried."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.agent.evaluation import write_json_atomic
from src.agent.provider_usage import normalize_usage


TERMINAL = {"completed", "failed", "expired", "cancelled"}


class BatchPending(RuntimeError):
    """The saved batch exists but is not ready to collect."""


class BatchInterventionRequired(RuntimeError):
    """Automatic progress would risk duplicate spend or corrupt evidence."""


def dspy_chat_body(program, inputs: dict, *, lm) -> dict:
    """Render the same DSPy ChatAdapter prompt for OpenAI's batch endpoint."""
    from dspy.adapters.chat_adapter import ChatAdapter

    model = str(lm.model).removeprefix("openai/")
    messages = ChatAdapter().format(program.signature, program.demos, inputs)
    kwargs = dict(lm.kwargs)
    max_tokens = int(kwargs.get("max_tokens", 4096))
    modern = model.startswith(("gpt-5", "o1", "o3", "o4"))
    body = {"model": model, "messages": messages,
            "max_completion_tokens" if modern else "max_tokens": max_tokens}
    if not modern and kwargs.get("temperature") is not None:
        body["temperature"] = kwargs["temperature"]
    if modern and kwargs.get("reasoning_effort"):
        body["reasoning_effort"] = kwargs["reasoning_effort"]
    return body


def parse_dspy_batch_body(program, body: dict):
    """Apply DSPy's normal typed parser to one successful Batch response."""
    from dspy.adapters.chat_adapter import ChatAdapter

    choice = body["choices"][0]
    message = choice["message"]
    if choice.get("finish_reason") == "length":
        raise ValueError("Batch completion exhausted its output-token allowance")
    if message.get("refusal"):
        raise ValueError("Batch completion was refused")
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Unexpected Batch finish reason: {choice.get('finish_reason')}")
    parsed = ChatAdapter().parse(program.signature, message.get("content") or "")
    return SimpleNamespace(**parsed)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _payload(jobs: list[dict]) -> bytes:
    ids = [job["custom_id"] for job in jobs]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Batch custom IDs must be nonempty and unique")
    lines = [{"custom_id": job["custom_id"], "method": "POST",
              "url": "/v1/chat/completions", "body": job["body"]} for job in jobs]
    return ("\n".join(json.dumps(line, sort_keys=True, separators=(",", ":")) for line in lines) + "\n").encode()


def submit_batch(path: Path, jobs: list[dict], *, client, metadata: dict) -> dict:
    """Submit once, checkpointing every boundary that could become ambiguous."""
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("status") == "submission_uncertain":
            raise BatchInterventionRequired("Batch submission is uncertain; adopt its provider ID explicitly")
        raise BatchInterventionRequired("A batch checkpoint already exists; collect or resolve it instead")
    payload = _payload(jobs)
    state = {"version": 1, "status": "uploading", "created_at": _now(),
             "payload_sha256": hashlib.sha256(payload).hexdigest(),
             "custom_ids": [job["custom_id"] for job in jobs], "metadata": metadata,
             "requests": {job["custom_id"]: {
                 "model": job["body"].get("model"),
                 "reserved_output_tokens": job["body"].get("max_completion_tokens", job["body"].get("max_tokens")),
             } for job in jobs},
             "results": {}, "errors": {}}
    write_json_atomic(path, state)
    try:
        uploaded = client.files.create(file=("deepswing-evaluations.jsonl", payload, "application/jsonl"),
                                       purpose="batch")
    except Exception as exc:
        state.update(status="upload_failed", error=str(exc)[:500], failed_at=_now())
        write_json_atomic(path, state)
        raise
    state.update(status="submitting", input_file_id=uploaded.id)
    write_json_atomic(path, state)
    try:
        remote = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                       completion_window="24h", metadata=metadata)
    except Exception as exc:
        state.update(status="submission_uncertain", error=str(exc)[:500], failed_at=_now())
        write_json_atomic(path, state)
        raise BatchInterventionRequired(
            "Batch submission outcome is uncertain; do not resubmit automatically"
        ) from exc
    state.update(status=_value(remote, "status", "validating"), batch_id=remote.id, submitted_at=_now())
    write_json_atomic(path, state)
    return state


def adopt_batch_id(path: Path, batch_id: str, *, authorized_by: str) -> dict:
    if not batch_id.strip() or not authorized_by.strip():
        raise ValueError("Adoption requires a batch ID and authorizer")
    state = json.loads(path.read_text())
    if state.get("status") not in ("submission_uncertain", "submitting") or state.get("batch_id"):
        raise ValueError("Only an uncertain submission without an ID can be adopted")
    state.update(status="adopted", batch_id=batch_id.strip(), adopted_at=_now(),
                 adopted_by=authorized_by.strip())
    write_json_atomic(path, state)
    return state


def collect_batch(path: Path, *, client) -> dict:
    """Retrieve and persist available results once, matching only expected IDs."""
    state = json.loads(path.read_text())
    batch_id = state.get("batch_id")
    if not batch_id:
        raise BatchInterventionRequired("Batch has no confirmed provider ID")
    remote = client.batches.retrieve(batch_id)
    if (_value(remote, "input_file_id") != state.get("input_file_id") or
            _value(remote, "metadata") != state.get("metadata")):
        raise ValueError("Remote batch identity does not match the durable checkpoint")
    status = _value(remote, "status")
    state["status"] = status
    state["checked_at"] = _now()
    write_json_atomic(path, state)
    if status not in TERMINAL:
        raise BatchPending(f"Batch {batch_id} is {status}")

    records = {}
    expected = set(state["custom_ids"])
    for file_id in (_value(remote, "output_file_id"), _value(remote, "error_file_id")):
        if not file_id:
            continue
        content = client.files.content(file_id)
        text = _value(content, "text", "")
        for line in text.splitlines():
            row = json.loads(line)
            custom_id = row.get("custom_id")
            if custom_id not in expected or custom_id in records:
                raise ValueError("Unexpected or duplicate Batch result ID")
            records[custom_id] = row
    for custom_id in state["custom_ids"]:
        if custom_id in state["results"] or custom_id in state["errors"]:
            continue
        row = records.get(custom_id)
        response = (row or {}).get("response") or {}
        body = response.get("body") or {}
        usage = normalize_usage("openai", body.get("usage"), model=body.get("model"))
        if not row or row.get("error") or response.get("status_code") != 200:
            state["errors"][custom_id] = {"error": (row or {}).get("error") or
                                         f"HTTP {response.get('status_code') or status}", "usage": usage}
        else:
            state["results"][custom_id] = {"body": body, "usage": usage}
        write_json_atomic(path, state)
    missing = expected - set(state["results"]) - set(state["errors"])
    if missing:
        state["status"] = "partial_failure"
        state["missing_ids"] = sorted(missing)
    elif state["errors"]:
        state["status"] = "partial_failure"
    else:
        state["status"] = "complete"
    state["collected_at"] = _now()
    write_json_atomic(path, state)
    return state
