from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from config.settings import settings
from src.agent.candidates import register_candidate
from src.agent.shadow import (
    authorize_shadow_retry,
    build_forward_evidence,
    collect_mature_shadow_outcomes,
    collect_shadow_decision,
)


@pytest.fixture
def shadow_env(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(settings, "shadow_enabled", True)
    monkeypatch.setattr(settings, "shadow_max_candidates_per_track", 1)
    monkeypatch.setattr(settings, "shadow_max_requests_per_day", 2)
    monkeypatch.setattr(settings, "shadow_output_tokens_per_request", 1000)
    monkeypatch.setattr(settings, "shadow_max_reserved_output_tokens_per_day", 32000)
    monkeypatch.setattr(settings, "shadow_max_input_bytes", 10000)
    source = tmp_path / "candidate.json"
    source.write_text(json.dumps({"instructions": "candidate"}))
    registered = register_candidate(
        "gpt", source, run_id="run-1", incumbent_hash="baseline",
        corpus_hash="corpus-1", gate={"promote": True},
    )
    candidate_hash = json.loads((registered / "manifest.json").read_text())["candidate_hash"]
    return SimpleNamespace(root=tmp_path, candidate_hash=candidate_hash)


def inputs():
    return {
        "technicals": "price 100 atr 2", "regime": "trending",
        "news_summary": "none", "macro_context": "neutral", "heuristics": "none",
    }


def incumbent():
    return {
        "action": "BUY", "confidence": 0.7, "stop_loss": 97.0, "target": 108.0,
        "reasoning": "incumbent", "program_hash": "incumbent-1",
    }


def prediction(action="PASS"):
    return SimpleNamespace(action=action, confidence=0.8, stop_loss=96.0, target=108.0,
                           reasoning="candidate")


def collect(stamp, predict):
    return collect_shadow_decision(
        track="gpt", ticker="AAPL", market="us", decision_time=stamp,
        price=100.0, atr=2.0, entry_inputs=inputs(), incumbent=incumbent(), predict=predict,
    )


def test_shadow_output_is_persisted_but_not_returned_as_an_execution_decision(shadow_env):
    case = collect(datetime(2026, 9, 1, 12), lambda *_: prediction())
    assert case["status"] == "awaiting_outcome"
    assert case["incumbent"]["action"] == "BUY"
    assert case["candidate"]["action"] == "PASS"
    assert case["candidate_hash"] == shadow_env.candidate_hash
    assert "executed" not in case["candidate"]


def test_exact_request_is_reused_without_another_attempt(shadow_env):
    calls = []
    stamp = datetime(2026, 9, 1, 12)
    first = collect(stamp, lambda *_: calls.append(1) or prediction())
    second = collect(stamp, lambda *_: calls.append(2) or prediction("BUY"))
    assert first == second
    assert calls == [1]
    ledger = json.loads((shadow_env.root / "shadow/gpt/ledger.json").read_text())
    assert len(next(iter(ledger["days"].values()))["attempts"]) == 1


def test_failed_request_is_retained_and_not_automatically_retried(shadow_env):
    stamp = datetime(2026, 9, 1, 12)
    failed = collect(stamp, lambda *_: (_ for _ in ()).throw(RuntimeError("provider failed")))
    assert failed["status"] == "failed"
    cached = collect(stamp, lambda *_: prediction())
    assert cached["status"] == "failed"
    assert cached["error"] == "provider failed"


def test_retry_requires_explicit_authorization_and_counts_again(shadow_env):
    stamp = datetime(2026, 9, 1, 12)
    failed = collect(stamp, lambda *_: (_ for _ in ()).throw(RuntimeError("provider failed")))
    authorize_shadow_retry(
        "gpt", failed["request_id"], authorized_by="alex", reason="transient provider outage",
    )
    retried = collect(stamp, lambda *_: prediction("PASS"))
    assert retried["status"] == "awaiting_outcome"
    assert retried["request"]["attempt"] == 2
    assert retried["previous_attempts"][0]["error"] == "provider failed"
    ledger = json.loads((shadow_env.root / "shadow/gpt/ledger.json").read_text())
    assert len(next(iter(ledger["days"].values()))["attempts"]) == 2


def test_daily_request_and_reservation_budget_stops_before_call(shadow_env):
    calls = []
    collect(datetime(2026, 9, 1, 12), lambda *_: calls.append(1) or prediction())
    collect(datetime(2026, 9, 1, 13), lambda *_: calls.append(2) or prediction())
    with pytest.raises(RuntimeError, match="daily request limit"):
        collect(datetime(2026, 9, 1, 14), lambda *_: calls.append(3) or prediction())
    assert calls == [1, 2]


def test_disabled_shadow_mode_performs_no_work(shadow_env, monkeypatch):
    monkeypatch.setattr(settings, "shadow_enabled", False)
    assert collect(datetime(2026, 9, 1, 12), lambda *_: pytest.fail("must not call")) is None
    assert not (shadow_env.root / "shadow").exists()


def test_mature_collector_freezes_one_path_and_scores_both_arms(shadow_env, monkeypatch):
    monkeypatch.setattr(settings, "counterfactual_horizon_days", 3)
    stamp = datetime(2026, 9, 1, 12)
    collect(stamp, lambda *_: prediction("PASS"))
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0], "High": [102.0, 107.0, 103.0],
            "Low": [99.0, 100.0, 101.0], "Close": [101.0, 106.0, 102.0],
        },
        index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
    )
    monkeypatch.setattr("src.data.market_data.fetch_ohlcv", lambda *args, **kwargs: frame)
    result = collect_mature_shadow_outcomes("gpt", now=stamp + timedelta(days=5))
    assert result == {"completed": 1, "pending": 0, "failed": 0}
    case_path, = (shadow_env.root / "shadow/gpt/cases").glob("*.json")
    case = json.loads(case_path.read_text())
    assert case["status"] == "complete"
    assert case["outcomes"]["incumbent"]["executed"] is True
    assert case["outcomes"]["candidate"]["executed"] is False
    assert case["path"]["bars"][0]["date"] == "2026-09-02"
    assert case["session_coverage"]["expected_sessions"] == ["2026-09-02", "2026-09-03", "2026-09-04"]


def test_forward_evidence_counts_non_overlapping_periods_not_raw_days(shadow_env, monkeypatch):
    monkeypatch.setattr(settings, "counterfactual_horizon_days", 14)
    monkeypatch.setattr(settings, "shadow_min_completed_cases", 3)
    monkeypatch.setattr(settings, "shadow_min_non_overlapping_periods", 2)
    monkeypatch.setattr(settings, "shadow_min_mean_score_gain", 0.01)
    cases = shadow_env.root / "shadow/gpt/cases"
    cases.mkdir(parents=True)
    for i, day in enumerate((1, 5, 20), 1):
        payload = {
            "candidate_hash": shadow_env.candidate_hash, "status": "complete",
            "decision_time": datetime(2026, 1, day, 12).isoformat(),
            "horizon_days": 14,
            "outcomes": {
                "incumbent": {"score": 0.4, "net_r": -0.2},
                "candidate": {"score": 0.6, "net_r": 0.4},
            },
        }
        (cases / f"case-{i}.json").write_text(json.dumps(payload))
    report = build_forward_evidence("gpt", shadow_env.candidate_hash)
    assert report["completed_cases"] == 3
    assert report["non_overlapping_periods"] == 2
    assert report["mean_score_gain"] == pytest.approx(0.2)
    assert report["eligible_for_review"] is True
    assert (shadow_env.root / "candidates/gpt" / shadow_env.candidate_hash / "forward_evidence.json").exists()


def test_daily_collection_accumulates_independent_windows(shadow_env):
    cases = shadow_env.root / "shadow/gpt/cases"
    cases.mkdir(parents=True)
    for i in range(60):
        (cases / f"{i}.json").write_text(json.dumps({
            "candidate_hash": shadow_env.candidate_hash, "status": "complete",
            "decision_time": (datetime(2026, 1, 1) + timedelta(days=i)).isoformat(),
            "horizon_days": 14,
            "outcomes": {"candidate": {"score": .8, "net_r": 1},
                         "incumbent": {"score": .5, "net_r": 0}},
        }))
    report = build_forward_evidence("gpt", shadow_env.candidate_hash)
    assert report["non_overlapping_periods"] == 4
    assert report["scored_cases"] == 4
    assert report["overlapping_cases_excluded"] == 56


def test_truncated_outcome_history_remains_pending(shadow_env, monkeypatch):
    stamp = datetime(2026, 9, 1)
    collect(stamp, lambda *_: prediction())
    frame = pd.DataFrame({"Open": [100]*3, "High": [101]*3, "Low": [99]*3, "Close": [100]*3},
                         index=pd.date_range("2026-09-02", periods=3))
    monkeypatch.setattr("src.data.market_data.fetch_ohlcv", lambda *a, **kw: frame)
    summary = collect_mature_shadow_outcomes("gpt", now=stamp + timedelta(days=20))
    assert summary["pending"] == 1 and summary["completed"] == 0


def test_gpt_reserves_effective_output_allowance(shadow_env):
    case = collect(datetime(2026, 9, 1), lambda *_: prediction())
    assert case["request"]["reserved_output_tokens"] == 16000
