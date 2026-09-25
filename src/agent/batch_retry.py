"""Explicit, cumulative-budget retries; original Batch responses remain immutable."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.agent.evaluation import write_json_atomic
from src.agent.openai_batch import BatchPending, BatchInterventionRequired, collect_batch, submit_batch


def merged_batch(run_dir: Path, *, client=None) -> dict:
    """Read/collect submitted retry batches; this function cannot submit work."""
    original = json.loads((run_dir / "batch.json").read_text())
    merged = {**original, "results": dict(original.get("results", {})),
              "errors": dict(original.get("errors", {})), "changed_limits": False}
    from src.agent.batch_evaluation import digest
    for folder in sorted((run_dir / "retries").glob("*")):
        authorization = json.loads((folder / "authorization.json").read_text())
        checkpoint = folder / "batch.json"
        if not checkpoint.exists():
            raise BatchInterventionRequired(f"Interrupted retry preparation: {folder}")
        retry = json.loads(checkpoint.read_text())
        expected_metadata = {"run": original["metadata"]["run"],
                             "plan_hash": original["metadata"]["plan_hash"],
                             "retry_hash": digest(authorization)}
        if retry["metadata"] != expected_metadata:
            raise ValueError("Retry authorization does not match checkpoint")
        if not retry.get("batch_id"):
            raise BatchInterventionRequired(f"Retry requires manual ID recovery: {checkpoint}")
        if retry["status"] not in ("complete", "partial_failure"):
            if client is None:
                raise BatchPending(f"Retry is pending: {checkpoint}")
            retry = collect_batch(checkpoint, client=client)
        if not set(retry["custom_ids"]).issubset(original["custom_ids"]):
            raise ValueError("Retry contains unrelated request IDs")
        for ident in retry["custom_ids"]:
            merged["results"].pop(ident, None)
            merged["errors"].pop(ident, None)
        merged["results"].update(retry["results"])
        merged["errors"].update(retry["errors"])
        merged["changed_limits"] |= authorization["changed_limits"]
    if merged["status"] in ("complete", "partial_failure"):
        merged["status"] = "partial_failure" if merged["errors"] else "complete"
    return merged


def submit_failed_retry(run_dir: Path, *, authorized_by: str, reason: str, client,
                        output_tokens: int | None = None, max_requests: int | None = None,
                        max_reserved_output_tokens: int | None = None) -> Path:
    """Retry only failed requests after explicit authorization, never whole arms."""
    import fcntl
    run_dir = Path(run_dir)
    with (run_dir / ".retry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _submit(run_dir, authorized_by, reason, client, output_tokens,
                       max_requests, max_reserved_output_tokens)


def _submit(run_dir, authorized_by, reason, client, output_tokens, max_requests, max_tokens):
    import dspy
    from src.agent.decision import TradeDecision
    from src.agent.compiled_program import program_fingerprint
    from src.agent.bounded_search import SearchBudget
    from src.agent.batch_evaluation import digest
    from src.agent.openai_batch import parse_dspy_batch_body
    if not authorized_by.strip() or not reason.strip():
        raise ValueError("Retry requires an authorizer and reason")
    report = json.loads((run_dir / "report.json").read_text())
    if report["status"] in ("rejected", "pending_forward_evaluation", "historical_review_required"):
        raise ValueError("Finished experiments cannot be retried")
    plan = json.loads((run_dir / "batch_plan.json").read_text())
    fingerprint = plan.pop("hash")
    if digest(plan) != fingerprint or report["batch_plan_hash"] != fingerprint:
        raise ValueError("Frozen Batch plan changed")
    batch = merged_batch(run_dir)
    if batch["status"] not in ("complete", "partial_failure"):
        raise BatchInterventionRequired("Collect the original batch before retrying")
    failed = set(batch["errors"])
    for arm in ("incumbent", "candidate"):
        expected = plan["candidate_hash"] if arm == "candidate" else plan["incumbent_artifact_hash"]
        if program_fingerprint(run_dir / f"{arm}.json") != expected:
            raise ValueError("Frozen program changed")
        program = dspy.Predict(TradeDecision)
        program.load(str(run_dir / f"{arm}.json"))
        for ident in plan["mapping"][arm]:
            if ident in batch["results"]:
                try:
                    parse_dspy_batch_body(program, batch["results"][ident]["body"])
                except Exception:
                    failed.add(ident)
    if not failed:
        raise ValueError("No failed requests to retry")
    jobs = [json.loads(json.dumps(job)) for job in plan["jobs"] if job["custom_id"] in failed]
    changed = False
    for job in jobs:
        body = job["body"]
        key = "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        if output_tokens is not None:
            if type(output_tokens) is not int or output_tokens < body[key]:
                raise ValueError("Retry output allowance must not decrease")
            changed |= body[key] != output_tokens
            body[key] = output_tokens
    limits = dict(report["budget_limits"])
    if max_requests is not None:
        limits["requests"] = max_requests
    if max_tokens is not None:
        limits["reserved_output_tokens"] = max_tokens
    budget = SearchBudget(run_dir / "budget.json", max_requests=limits["requests"],
                          max_input_bytes=limits["input_bytes"],
                          max_reserved_output_tokens=limits["reserved_output_tokens"])
    # Preflight the group before recording or reserving any part of it.
    inputs = sum(len(json.dumps(j["body"]).encode()) for j in jobs)
    outputs = sum(j["body"].get("max_completion_tokens", j["body"].get("max_tokens")) for j in jobs)
    if (budget.state["requests"] + len(jobs) > limits["requests"] or
            budget.state["input_bytes"] + inputs > limits["input_bytes"] or
            budget.state["reserved_output_tokens"] + outputs > limits["reserved_output_tokens"]):
        raise ValueError("Cumulative retry budget exhausted; explicit higher limits required")
    folders = list((run_dir / "retries").glob("*"))
    folder = run_dir / "retries" / f"{len(folders)+1:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    authorization = {"authorized_by": authorized_by.strip(), "reason": reason.strip(),
                     "authorized_at": datetime.now(timezone.utc).isoformat(),
                     "jobs": jobs, "limits": limits, "changed_limits": changed}
    write_json_atomic(folder / "authorization.json", authorization)
    for job in jobs:
        body = job["body"]
        budget.reserve(len(json.dumps(body).encode()), body.get("max_completion_tokens", body.get("max_tokens")))
    report.update(status="batch_retry_pending", budget_limits=limits)
    write_json_atomic(run_dir / "report.json", report)
    submit_batch(folder / "batch.json", jobs, client=client,
                 metadata={**batch["metadata"], "retry_hash": digest(authorization)})
    return folder / "batch.json"
