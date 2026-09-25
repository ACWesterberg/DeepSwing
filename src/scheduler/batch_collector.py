"""Retrieval-only scheduled collection for already-submitted optimizer batches."""
from __future__ import annotations

import json
import logging

from config.settings import settings
from src.agent.evaluation import write_json_atomic
from src.agent.openai_batch import (
    BatchInterventionRequired, BatchPending, collect_batch,
)

logger = logging.getLogger(__name__)


def _client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key, max_retries=0)


def collect_existing_optimizer_batches(*, client=None) -> dict:
    """Collect confirmed batches only; never submit, adopt, or retry anything."""
    summary = {"checked": 0, "pending": 0, "completed": 0, "intervention": 0,
               "events": []}
    root = settings.compiled_dir / "evaluations"
    if not root.exists():
        return summary
    batch_client = client
    for path in sorted(root.glob("*/*/batch.json")):
        state = json.loads(path.read_text())
        summary["checked"] += 1
        prior = state.get("collector_reported_status")
        status = state.get("status")
        event = None
        if status == "submission_uncertain" or not state.get("batch_id"):
            summary["intervention"] += 1
            event = {"kind": "intervention", "path": str(path), "status": status,
                     "message": "Optimizer Batch needs manual ID recovery; it was not resubmitted."}
        elif status in ("complete", "partial_failure"):
            summary["completed"] += 1
            event = {"kind": "complete" if status == "complete" else "intervention",
                     "path": str(path), "status": status,
                     "message": ("Optimizer Batch results collected; final scoring can resume."
                                 if status == "complete" else
                                 "Optimizer Batch has partial failures; successful results were retained.")}
        else:
            if batch_client is None:
                batch_client = _client()
            try:
                state = collect_batch(path, client=batch_client)
                status = state["status"]
                if status in ("complete", "partial_failure"):
                    summary["completed"] += 1
                    event = {"kind": "complete" if status == "complete" else "intervention",
                             "path": str(path), "status": status,
                             "message": ("Optimizer Batch results collected; final scoring can resume."
                                         if status == "complete" else
                                         "Optimizer Batch has partial failures; successful results were retained.")}
            except BatchPending:
                summary["pending"] += 1
            except BatchInterventionRequired as exc:
                summary["intervention"] += 1
                event = {"kind": "intervention", "path": str(path), "status": status,
                         "message": str(exc)}
            except Exception as exc:
                summary["intervention"] += 1
                event = {"kind": "intervention", "path": str(path), "status": "collector_error",
                         "message": f"Optimizer Batch collection failed: {exc}"}
        if status in ("complete", "partial_failure") and (path.parent / "retries").exists():
            from src.agent.batch_retry import merged_batch
            try:
                if batch_client is None:
                    batch_client = _client()
                state = merged_batch(path.parent, client=batch_client)
                status = state["status"]
                event = None
                if status == "partial_failure":
                    event = {"kind": "intervention", "path": str(path), "status": "retry_failed",
                             "message": "Optimizer retry has failures; saved successes remain available."}
            except BatchPending:
                summary["pending"] += 1
                status, event = "retry_pending", None
            except Exception as exc:
                status = "retry_intervention"
                event = {"kind": "intervention", "path": str(path), "status": status, "message": str(exc)}
        if status == "complete" and (path.parent / "batch_plan.json").exists():
            try:
                from src.agent.batch_evaluation import finalize_batch_evaluation
                report = finalize_batch_evaluation(path.parent)
                event = {"kind": "complete", "path": str(path), "status": report["status"],
                         "message": f"Optimizer Batch scoring finished: {report['status']}."}
            except Exception as exc:
                event = {"kind": "intervention", "path": str(path), "status": "batch_scoring_error",
                         "message": f"Optimizer Batch scoring needs attention: {exc}"}
        if event and prior != event["status"]:
            summary["events"].append(event)
    return summary


def scheduled_batch_collection() -> dict:
    """Collect and notify only on completion or required intervention."""
    from src.notify.telegram import send_telegram

    summary = collect_existing_optimizer_batches()
    for event in summary["events"]:
        if send_telegram(f"DeepSwing: {event['message']}\n{event['path']}"):
            from pathlib import Path
            path = Path(event["path"])
            latest = json.loads(path.read_text())
            latest["collector_reported_status"] = event["status"]
            write_json_atomic(path, latest)
    if summary["events"]:
        logger.info("Optimizer Batch collection: %s", summary)
    return summary
