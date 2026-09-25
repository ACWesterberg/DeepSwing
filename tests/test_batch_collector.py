from __future__ import annotations

import json
from types import SimpleNamespace

from config.settings import settings
from src.scheduler.batch_collector import collect_existing_optimizer_batches


def write_state(tmp_path, monkeypatch, state):
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    path = tmp_path / "evaluations/gpt/run/batch.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(state))
    return path


def test_collector_is_quiet_for_no_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    assert collect_existing_optimizer_batches()["checked"] == 0


def test_uncertain_submission_never_constructs_client(tmp_path, monkeypatch):
    path = write_state(tmp_path, monkeypatch, {"status": "submission_uncertain"})
    summary = collect_existing_optimizer_batches(client=object())
    assert summary["intervention"] == 1 and len(summary["events"]) == 1
    assert "collector_reported_status" not in json.loads(path.read_text())
    assert collect_existing_optimizer_batches(client=object())["events"]


def test_pending_status_is_quiet(tmp_path, monkeypatch):
    write_state(tmp_path, monkeypatch, {"status": "in_progress", "batch_id": "batch-1",
        "input_file_id": "file-in", "metadata": {"run": "one"}, "custom_ids": [],
        "results": {}, "errors": {}})
    remote = SimpleNamespace(id="batch-1", status="in_progress", input_file_id="file-in",
        metadata={"run": "one"}, output_file_id=None, error_file_id=None)
    client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _: remote))
    summary = collect_existing_optimizer_batches(client=client)
    assert summary["pending"] == 1 and summary["events"] == []


def test_complete_status_emits_once_without_remote_call(tmp_path, monkeypatch):
    path = write_state(tmp_path, monkeypatch, {"status": "complete", "batch_id": "batch-1"})
    summary = collect_existing_optimizer_batches(client=object())
    assert summary["completed"] == 1 and summary["events"][0]["kind"] == "complete"
    assert collect_existing_optimizer_batches(client=object())["events"]
    assert "collector_reported_status" not in json.loads(path.read_text())


def test_failed_delivery_is_retried_until_acknowledged(tmp_path, monkeypatch):
    from src.scheduler.batch_collector import scheduled_batch_collection
    path = write_state(tmp_path, monkeypatch, {"status": "complete", "batch_id": "batch-1"})
    delivered = []
    monkeypatch.setattr("src.notify.telegram.send_telegram", lambda text: delivered.append(text) or False)
    scheduled_batch_collection()
    assert "collector_reported_status" not in json.loads(path.read_text())
    monkeypatch.setattr("src.notify.telegram.send_telegram", lambda text: delivered.append(text) or True)
    scheduled_batch_collection()
    scheduled_batch_collection()
    assert len(delivered) == 2
