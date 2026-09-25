from __future__ import annotations

import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest


def test_real_sdk_in_fresh_process(tmp_path):
    try:
        version("dspy")
    except PackageNotFoundError:
        pytest.skip("Install dspy-ai to run the real SDK smoke test")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _replay_with_real_dspy_and_local_dummy_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    def no_network(*args, **kwargs):
        raise AssertionError("SDK integration test must not use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)

    import dspy
    from dspy.utils import DummyLM
    from config.settings import settings
    from scripts.replay_decisions import _load_program
    from src.agent import decision
    from src.agent.compiled_program import save_compiled_program
    from src.agent.replay import ReplayExample, dspy_program, score_program

    monkeypatch.setattr(settings, "openai_api_key", "local-test-only")
    models = []

    def local_model(*args, **kwargs):
        lm = DummyLM([dict(action="BUY", confidence=0.8, stop_loss=97.0,
                           target=110.0, reasoning="test")])
        models.append(lm)
        return lm

    monkeypatch.setattr(decision, "build_lm", local_model)
    example = ReplayExample("gpt", "TEST", "us", "2026-08-01", {
        "technicals": "Price 100, ATR 2", "regime": "trending",
        "news_summary": "none", "macro_context": "none", "heuristics": "none",
        "heuristic_ids": ["not-a-model-input"],
    }, "BUY", 2.0)
    baseline = _load_program("baseline", "gpt")
    assert baseline.get_lm() is models[-1]
    assert score_program([example], dspy_program(baseline)).buys == 1

    path = tmp_path / "gpt_trade_decision.json"
    artifact = dspy.Predict(decision.TradeDecision)
    save_compiled_program(artifact, path, lambda p: dspy.Predict(decision.TradeDecision).load(str(p)))
    loaded = _load_program(str(path), "gpt")
    assert loaded.get_lm() is models[-1]
    assert score_program([example], dspy_program(loaded)).buys == 1

    # Exercise temporal metadata on actual DSPy Examples, not unit-suite mocks.
    from datetime import datetime
    from src.agent.evaluation import evaluate_program, corpus_fingerprint
    from src.scheduler.optimizer import _date_example, _make_example
    dated = _date_example(
        _make_example(example.entry_inputs, "BUY", 2.0),
        decision_time=datetime(2026, 8, 1), available_at=datetime(2026, 8, 15),
        ticker="TEST", market="us", source="real", example_id="sdk:1",
    )
    assert "decision_time" not in dated.inputs()
    assert corpus_fingerprint([dated.toDict()])
    loaded.set_lm(local_model())
    assert evaluate_program([dated], loaded)["buys"] == 1

    from src.agent.plan_replay import freeze_path, score_plans, plan_metric
    import pandas as pd
    frame = pd.DataFrame({"Open": [100.0], "High": [111.0], "Low": [99.0], "Close": [108.0]},
                         index=pd.to_datetime(["2026-08-02"]))
    example.plan_path = freeze_path(frame, 100.0, 2.0, "us", datetime(2026, 8, 1))
    loaded.set_lm(local_model())
    assert score_plans([example], loaded)["buys"] == 1
    from src.agent.portfolio_replay import PortfolioPolicy, score_portfolio
    loaded.set_lm(local_model())
    portfolio_report = score_portfolio([example], loaded, policy=PortfolioPolicy.current())
    assert portfolio_report["trades"] == 1
    assert portfolio_report["final_equity"] > portfolio_report["initial_equity"]
    dated["plan_path"] = example.plan_path
    stripped = _make_example(dated, "BUY", 2.0)
    assert "plan_path" not in stripped.toDict()
    assert plan_metric([dated])(stripped, dspy.Prediction(action="BUY", confidence=0.8,
                                                        stop_loss=97.0, target=110.0)) > 0.5

    # Batch rendering/parsing must preserve the exact DSPy signature contract
    # without making a provider request.
    from types import SimpleNamespace
    from src.agent.openai_batch import dspy_chat_body, parse_dspy_batch_body
    batch_lm = SimpleNamespace(model="openai/gpt-5", kwargs={
        "max_tokens": 16000, "reasoning_effort": "low", "temperature": 1.0,
    })
    body = dspy_chat_body(artifact, example.entry_inputs, lm=batch_lm)
    assert body["model"] == "gpt-5" and body["max_completion_tokens"] == 16000
    assert body["reasoning_effort"] == "low" and "temperature" not in body
    parsed = parse_dspy_batch_body(artifact, {"choices": [{"finish_reason": "stop", "message": {
        "content": "[[ ## action ## ]]\nBUY\n\n[[ ## confidence ## ]]\n0.8\n\n"
                   "[[ ## stop_loss ## ]]\n97.0\n\n[[ ## target ## ]]\n110.0\n\n"
                   "[[ ## reasoning ## ]]\ntest\n\n[[ ## completed ## ]]"
    }}]})
    assert parsed.action == "BUY" and parsed.confidence == .8
    from src.agent.single_request import single_predict
    class MalformedLM:
        num_retries = 3
        cache = True
        calls = 0
        def __call__(self, **kwargs):
            self.calls += 1
            assert self.num_retries == 0 and not self.cache
            return ["malformed response"]
    malformed = MalformedLM()
    with pytest.raises(Exception):
        single_predict(artifact, malformed, example.entry_inputs)
    assert malformed.calls == 1

    # Exercise submission -> out-of-order collection -> local scoring with the
    # real DSPy parser and a provider stub. No sockets are permitted above.
    import json
    from src.agent.batch_evaluation import prepare_batch_evaluation, finalize_batch_evaluation
    from src.agent.bounded_search import SearchBudget
    from src.agent.compiled_program import program_fingerprint
    from src.agent.openai_batch import collect_batch
    run_dir = tmp_path / "batch-run"
    run_dir.mkdir()
    for name in ("incumbent", "candidate"):
        artifact.save(str(run_dir / f"{name}.json"))
    report = {"status": "evaluating", "track": "gpt", "run_id": "sdk-batch",
              "candidate_hash": program_fingerprint(run_dir / "candidate.json"),
              "incumbent_hash": "baseline", "corpus_hash": "test-corpus",
              "thresholds": {"promotion_min_metric_gain": .01, "promotion_min_buys": 1,
                             "promotion_bootstrap_samples": 100}}
    submitted = []
    def upload(**kwargs):
        submitted.extend(json.loads(line) for line in kwargs["file"][1].decode().splitlines())
        return SimpleNamespace(id="file-test")
    remote = SimpleNamespace(id="batch-test", status="validating", input_file_id="file-test",
                             output_file_id="output-test", error_file_id=None)
    def create(**kwargs):
        remote.metadata = kwargs["metadata"]
        return remote
    content = ("[[ ## action ## ]]\nBUY\n[[ ## confidence ## ]]\n0.8\n"
               "[[ ## stop_loss ## ]]\n97\n[[ ## target ## ]]\n110\n[[ ## reasoning ## ]]\ntest")
    def download(_):
        return SimpleNamespace(text="\n".join(json.dumps({"custom_id": row["custom_id"],
            "response": {"status_code": 200, "body": {"choices": [{"finish_reason": "stop",
                "message": {"content": content}}]}}}) for row in reversed(submitted)))
    client = SimpleNamespace(files=SimpleNamespace(create=upload, content=download),
                             batches=SimpleNamespace(create=create, retrieve=lambda _: remote))
    rows = [{**dated.toDict(), "technicals": f"case-{i}", "ticker": f"T{i}",
             "decision_time": f"2026-08-{i+1:02d}T00:00:00"} for i in range(3)]
    budget = SearchBudget(run_dir / "budget.json", max_requests=6, max_input_bytes=1000000,
                          max_reserved_output_tokens=96000)
    prepare_batch_evaluation(run_dir, rows, (("incumbent", artifact), ("candidate", artifact)),
                             batch_lm, budget, report, client=client)
    assert len(submitted) == 3  # Identical arms share exact requests.
    remote.status = "completed"
    collect_batch(run_dir / "batch.json", client=client)
    finalized = finalize_batch_evaluation(run_dir)
    assert finalized["status"] == "rejected"  # Identical arms cannot improve.
    assert finalized["results"]["candidate"] == finalized["results"]["incumbent"]
    assert finalize_batch_evaluation(run_dir) == finalized

    # A paid failed completion is retained and only that request is retried.
    # Raising its allowance prevents automatic candidate registration.
    import shutil
    from src.agent.batch_retry import submit_failed_retry, merged_batch
    retry_run = tmp_path / "batch-retry-run"
    shutil.copytree(run_dir, retry_run)
    retry_report = json.loads((retry_run / "report.json").read_text())
    retry_report.update(status="batch_scoring_error", budget_limits={
        "requests": 6, "input_bytes": 1000000, "reserved_output_tokens": 96000})
    (retry_run / "report.json").write_text(json.dumps(retry_report))
    original = json.loads((retry_run / "batch.json").read_text())
    failed_id = original["custom_ids"][0]
    original["results"][failed_id]["body"]["choices"][0]["finish_reason"] = "length"
    (retry_run / "batch.json").write_text(json.dumps(original))
    submitted.clear()
    retry_path = submit_failed_retry(retry_run, authorized_by="sdk-test", reason="output exhausted",
                                     output_tokens=32000, client=client)
    assert len(submitted) == 1 and submitted[0]["custom_id"] == failed_id
    remote.status = "completed"
    collect_batch(retry_path, client=client)
    assert len(merged_batch(retry_run)["results"]) == 3
    retry_final = finalize_batch_evaluation(retry_run)
    assert retry_final["status"] == "historical_review_required"
    assert retry_final["evaluation_limits_changed"] is True


def _replay_missing_credentials_fails_before_model_creation(monkeypatch):
    from config.settings import settings
    from scripts.replay_decisions import _load_program
    import pytest

    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(ValueError, match="No API key"):
        _load_program("baseline", "gpt")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with pytest.MonkeyPatch.context() as monkeypatch:
        _replay_with_real_dspy_and_local_dummy_model(Path(sys.argv[1]), monkeypatch)
        _replay_missing_credentials_fails_before_model_creation(monkeypatch)
