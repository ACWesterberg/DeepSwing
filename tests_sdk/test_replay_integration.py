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
