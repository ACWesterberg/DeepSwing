from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.agent.openai_batch import (
    BatchInterventionRequired, BatchPending, adopt_batch_id, collect_batch,
    submit_batch,
)


class Files:
    def __init__(self):
        self.contents = {}
    def create(self, **kwargs):
        return SimpleNamespace(id="file-in")
    def content(self, file_id):
        return SimpleNamespace(text=self.contents[file_id])


class Batches:
    def __init__(self):
        self.remote = SimpleNamespace(id="batch-1", status="validating", input_file_id="file-in",
                                      metadata={"run": "one"}, output_file_id=None, error_file_id=None)
        self.fail_create = False
    def create(self, **kwargs):
        if self.fail_create:
            raise RuntimeError("connection lost")
        return self.remote
    def retrieve(self, batch_id):
        assert batch_id == "batch-1"
        return self.remote


class Client:
    def __init__(self):
        self.files, self.batches = Files(), Batches()


def jobs():
    return [{"custom_id": "incumbent-0", "body": {"model": "gpt-test", "messages": []}},
            {"custom_id": "candidate-0", "body": {"model": "gpt-test", "messages": []}}]


def test_submit_saves_confirmed_identity_and_never_resubmits(tmp_path):
    path, client = tmp_path / "batch.json", Client()
    state = submit_batch(path, jobs(), client=client, metadata={"run": "one"})
    assert state["batch_id"] == "batch-1"
    with pytest.raises(BatchInterventionRequired, match="checkpoint"):
        submit_batch(path, jobs(), client=client, metadata={"run": "one"})


def test_uncertain_submission_requires_explicit_adoption(tmp_path):
    path, client = tmp_path / "batch.json", Client()
    client.batches.fail_create = True
    with pytest.raises(BatchInterventionRequired, match="uncertain"):
        submit_batch(path, jobs(), client=client, metadata={"run": "one"})
    assert json.loads(path.read_text())["status"] == "submission_uncertain"
    adopted = adopt_batch_id(path, "batch-1", authorized_by="alex")
    assert adopted["status"] == "adopted"


def test_pending_collection_is_read_only_except_checkpoint(tmp_path):
    path, client = tmp_path / "batch.json", Client()
    submit_batch(path, jobs(), client=client, metadata={"run": "one"})
    with pytest.raises(BatchPending, match="validating"):
        collect_batch(path, client=client)


def test_collection_matches_ids_and_preserves_partial_success(tmp_path):
    path, client = tmp_path / "batch.json", Client()
    submit_batch(path, jobs(), client=client, metadata={"run": "one"})
    good = {"custom_id": "incumbent-0", "response": {"status_code": 200, "body": {
        "model": "gpt-test", "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}}}
    bad = {"custom_id": "candidate-0", "error": {"message": "expired"}}
    client.files.contents["out"] = json.dumps(good)
    client.files.contents["err"] = json.dumps(bad)
    client.batches.remote = SimpleNamespace(id="batch-1", status="completed", input_file_id="file-in",
        metadata={"run": "one"}, output_file_id="out", error_file_id="err")
    state = collect_batch(path, client=client)
    assert state["status"] == "partial_failure"
    assert state["results"]["incumbent-0"]["usage"]["input_tokens"] == 8
    assert "candidate-0" in state["errors"]


def test_collection_rejects_unexpected_or_duplicate_ids(tmp_path):
    path, client = tmp_path / "batch.json", Client()
    submit_batch(path, jobs(), client=client, metadata={"run": "one"})
    client.files.contents["out"] = json.dumps({"custom_id": "stranger", "response": {"status_code": 200}})
    client.batches.remote = SimpleNamespace(id="batch-1", status="completed", input_file_id="file-in",
        metadata={"run": "one"}, output_file_id="out", error_file_id=None)
    with pytest.raises(ValueError, match="Unexpected"):
        collect_batch(path, client=client)


def test_process_crash_during_submission_allows_id_adoption(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"status": "submitting", "input_file_id": "file-in"}))
    assert adopt_batch_id(path, "batch-1", authorized_by="alex")["batch_id"] == "batch-1"
