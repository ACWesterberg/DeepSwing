import json
from types import SimpleNamespace

import pytest

from src.agent.batch_evaluation import digest
from src.agent.batch_retry import merged_batch, submit_failed_retry
from src.agent.compiled_program import program_fingerprint
from src.agent.evaluation import write_json_atomic
from src.agent.openai_batch import BatchInterventionRequired, BatchPending


@pytest.fixture
def run(tmp_path, monkeypatch):
    import dspy
    monkeypatch.setattr(dspy, "Predict", lambda _: SimpleNamespace(load=lambda _: None))
    monkeypatch.setattr("src.agent.openai_batch.parse_dspy_batch_body", lambda *args: None)
    for arm in ("incumbent", "candidate"):
        write_json_atomic(tmp_path / f"{arm}.json", {"instruction": arm})
    plan = {"jobs": [{"custom_id": ident, "body": {"model": "gpt-5", "messages": [],
                                                  "max_completion_tokens": 16000}} for ident in ("ok", "failed")],
            "mapping": {"incumbent": ["ok"], "candidate": ["failed"]},
            "candidate_hash": program_fingerprint(tmp_path / "candidate.json"),
            "incumbent_artifact_hash": program_fingerprint(tmp_path / "incumbent.json")}
    plan["hash"] = digest(plan)
    write_json_atomic(tmp_path / "batch_plan.json", plan)
    write_json_atomic(tmp_path / "report.json", {"status": "batch_pending", "batch_plan_hash": plan["hash"],
        "budget_limits": {"requests": 2, "input_bytes": 100000, "reserved_output_tokens": 32000}})
    write_json_atomic(tmp_path / "budget.json", {"version": 1, "requests": 2, "input_bytes": 1000,
                                              "reserved_output_tokens": 32000})
    write_json_atomic(tmp_path / "batch.json", {"status": "partial_failure", "batch_id": "original",
        "metadata": {"run": "run", "plan_hash": plan["hash"]}, "custom_ids": ["ok", "failed"],
        "results": {"ok": {"body": {"result": "preserved"}}}, "errors": {"failed": {"error": "expired"}}})
    return tmp_path


def test_exhausted_budget_stops_before_upload(run):
    with pytest.raises(ValueError, match="budget exhausted"):
        submit_failed_retry(run, authorized_by="alex", reason="expired", client=object())
    assert not (run / "retries").exists()
    assert json.loads((run / "budget.json").read_text())["requests"] == 2


def test_retry_sends_only_failure_retains_original_and_counts_cumulatively(run):
    sent = []
    def upload(**kwargs):
        sent.extend(json.loads(line) for line in kwargs["file"][1].decode().splitlines())
        return SimpleNamespace(id="file-retry")
    client = SimpleNamespace(files=SimpleNamespace(create=upload),
        batches=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="retry-1", status="validating")))
    original = (run / "batch.json").read_bytes()
    path = submit_failed_retry(run, authorized_by="alex", reason="expired", client=client,
                               max_requests=3, max_reserved_output_tokens=64000, output_tokens=32000)
    assert [r["custom_id"] for r in sent] == ["failed"]
    assert sent[0]["body"]["max_completion_tokens"] == 32000
    assert json.loads((run / "budget.json").read_text())["reserved_output_tokens"] == 64000
    with pytest.raises(BatchPending):
        submit_failed_retry(run, authorized_by="alex", reason="again", client=object())
    checkpoint = json.loads(path.read_text())
    checkpoint.update(status="complete", results={"failed": {"body": {"result": "retry"}}}, errors={})
    write_json_atomic(path, checkpoint)
    merged = merged_batch(run)
    assert merged["status"] == "complete" and merged["changed_limits"]
    assert merged["results"]["ok"]["body"]["result"] == "preserved"
    assert (run / "batch.json").read_bytes() == original
    with pytest.raises(ValueError, match="No failed requests"):
        submit_failed_retry(run, authorized_by="alex", reason="again", client=object())


def test_uncertain_retry_is_never_resubmitted(run):
    def lose_response(**kwargs):
        raise RuntimeError("connection lost")
    client = SimpleNamespace(files=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="file")),
                             batches=SimpleNamespace(create=lose_response))
    with pytest.raises(BatchInterventionRequired):
        submit_failed_retry(run, authorized_by="alex", reason="retry", client=client,
                             max_requests=3, max_reserved_output_tokens=48000)
    with pytest.raises(BatchInterventionRequired, match="ID recovery"):
        submit_failed_retry(run, authorized_by="alex", reason="again", client=object())
