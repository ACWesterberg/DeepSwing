"""Frozen optimizer Batch plans and local-only final scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config.settings import settings
from src.agent.compiled_program import BASELINE, program_fingerprint
from src.agent.evaluation import promotion_decision, summarize_actions, write_json_atomic
from src.agent.openai_batch import dspy_chat_body, parse_dspy_batch_body, submit_batch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_batch_evaluation(run_dir, examples, programs, lm, budget, report, *, client):
    from src.agent.replay import DECISION_INPUTS

    rows = [e.toDict() if hasattr(e, "toDict") else dict(e) for e in examples]
    jobs, mapping = {}, {}
    for name, program in programs:
        mapping[name] = []
        for row in rows:
            body = dspy_chat_body(program, {k: row[k] for k in DECISION_INPUTS}, lm=lm)
            ident = digest(body)
            jobs[ident] = {"custom_id": ident, "body": body}
            mapping[name].append(ident)
    plan = {"version": 1, "examples": rows, "mapping": mapping,
            "jobs": list(jobs.values()), "candidate_hash": report["candidate_hash"],
            "incumbent_artifact_hash": program_fingerprint(run_dir / "incumbent.json"),
            "thresholds": report["thresholds"], "plan_mode": any(e.get("plan_path") for e in rows)}
    plan["hash"] = digest(plan)
    write_json_atomic(run_dir / "batch_plan.json", plan)
    # Reserve the entire group before upload. A budget failure cannot send a
    # partial experiment; reservations remain spent after uncertain submission.
    for job in jobs.values():
        body = job["body"]
        budget.reserve(len(json.dumps(body).encode()),
                       body.get("max_completion_tokens", body.get("max_tokens")))
    report.update(status="batch_pending", batch_plan_hash=plan["hash"])
    write_json_atomic(run_dir / "report.json", report)
    return submit_batch(run_dir / "batch.json", list(jobs.values()), client=client,
                        metadata={"run": report["run_id"], "plan_hash": plan["hash"]})


def finalize_batch_evaluation(run_dir: Path) -> dict:
    """Score saved results with frozen thresholds. Performs no remote calls."""
    import dspy
    from src.agent.decision import TradeDecision
    from src.agent.candidates import register_candidate
    from src.agent.plan_replay import evaluate_plan, summarize_plans

    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text())
    if report["status"] in ("rejected", "pending_forward_evaluation", "historical_review_required"):
        return report
    try:
        plan = json.loads((run_dir / "batch_plan.json").read_text())
        saved_hash = plan.pop("hash")
        if digest(plan) != saved_hash or saved_hash != report["batch_plan_hash"]:
            raise ValueError("Frozen Batch plan changed")
        from src.agent.batch_retry import merged_batch
        batch = merged_batch(run_dir)
        if batch["status"] != "complete" or batch.get("errors"):
            raise ValueError("Batch evaluation is incomplete; successes remain saved")
        if batch["metadata"] != {"run": report["run_id"], "plan_hash": saved_hash}:
            raise ValueError("Batch does not match the evaluation plan")
        expected_ids = {job["custom_id"] for job in plan["jobs"]}
        if set(batch["results"]) != expected_ids:
            raise ValueError("Batch result IDs do not match the evaluation plan")
        for name, expected in (("candidate", plan["candidate_hash"]),
                               ("incumbent", plan["incumbent_artifact_hash"])):
            if program_fingerprint(run_dir / f"{name}.json") != expected:
                raise ValueError(f"Frozen {name} artifact changed")
        rows = plan["examples"]
        results = {}
        from src.agent.plan_replay import evaluate_plan_examples
        for name in ("always_buy", "always_pass"):
            results[name] = (evaluate_plan_examples(rows, reference=name) if plan["plan_mode"] else
                             summarize_actions(rows, ["BUY" if name == "always_buy" else "PASS"] * len(rows)))
        predictions = {}
        for name in ("incumbent", "candidate"):
            program = dspy.Predict(TradeDecision)
            program.load(str(run_dir / f"{name}.json"))
            outputs = [parse_dspy_batch_body(program, batch["results"][ident]["body"])
                       for ident in plan["mapping"][name]]
            predictions[name] = [vars(prediction) for prediction in outputs]
            results[name] = (summarize_plans([evaluate_plan(row["plan_path"], prediction)
                                            for row, prediction in zip(rows, outputs)])
                             if plan["plan_mode"] else summarize_actions(rows, [p.action for p in outputs]))
        thresholds = plan["thresholds"]
        gate = promotion_decision(rows, results, min_gain=thresholds["promotion_min_metric_gain"],
                                  min_buys=thresholds["promotion_min_buys"],
                                  bootstrap_samples=thresholds["promotion_bootstrap_samples"],
                                  required_references={"candidate", "incumbent", "always_buy", "always_pass"})
        report.update(results=results, predictions=predictions, gate=gate, status="rejected")
        report["evaluation_limits_changed"] = batch.get("changed_limits", False)
        if batch.get("changed_limits"):
            report["status"] = "historical_review_required"
        elif gate["promote"]:
            active = settings.compiled_dir / f"{report['track']}_trade_decision.json"
            current = program_fingerprint(active)
            if (active.exists() and current is None) or (current or BASELINE) != report["incumbent_hash"]:
                raise ValueError("Active incumbent changed during Batch evaluation")
            registered = register_candidate(report["track"], run_dir / "candidate.json",
                run_id=report["run_id"], incumbent_hash=report["incumbent_hash"],
                corpus_hash=report["corpus_hash"], gate=gate)
            report.update(status="pending_forward_evaluation", candidate_registry_path=str(registered))
        report.pop("error", None)
        write_json_atomic(report_path, report)
        return report
    except Exception as exc:
        report.update(status="batch_scoring_error", error=str(exc))
        write_json_atomic(report_path, report)
        raise
