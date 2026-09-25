from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from config.settings import settings
from src.agent.bounded_search import (
    SearchBudget,
    authorize_search_retry,
    compact_training_summary,
    exact_request,
    select_screen_examples,
)
from src.agent.provider_usage import normalize_usage, usage_report


@pytest.fixture
def search_env(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    return tmp_path


def budget(tmp_path, requests=3):
    return SearchBudget(tmp_path / "budget.json", max_requests=requests,
                        max_input_bytes=10000, max_reserved_output_tokens=3000)


def test_training_summary_is_aggregate_and_excludes_model_inputs_and_paths():
    rows = [{
        "action": "BUY", "source": "real", "market": "us", "r_multiple": 1.5,
        "technicals": "secret row", "plan_path": {"future": "must not leak"},
    }, {
        "action": "PASS", "source": "counterfactual", "market": "nordic", "r_multiple": -1.0,
        "technicals": "another row", "plan_path": {"future": "must not leak"},
    }]
    summary = compact_training_summary(rows)
    assert json.loads(summary)["examples"] == 2
    assert "secret row" not in summary
    assert "future" not in summary


def test_screen_selection_spreads_chronologically_without_reading_outcomes():
    start = datetime(2026, 1, 1)
    rows = [{"decision_time": (start + timedelta(days=i)).isoformat(), "r_multiple": i} for i in range(20)]
    selected = select_screen_examples(rows, 4)
    assert [row["r_multiple"] for row in selected] == [0, 6, 13, 19]


def test_exact_completed_request_is_reused_without_budget_or_call(search_env):
    calls = []
    first_budget = budget(search_env)
    first = exact_request("gpt", {"x": 1}, output_tokens=1000, budget=first_budget,
                          call=lambda: calls.append(1) or {"answer": "ok"})
    second_budget = budget(search_env)
    second = exact_request("gpt", {"x": 1}, output_tokens=1000, budget=second_budget,
                           call=lambda: calls.append(2) or {"answer": "wrong"})
    assert first == second == {"answer": "ok"}
    assert calls == [1]
    assert second_budget.state["requests"] == 1


def test_failed_request_requires_explicit_retry(search_env):
    payload = {"kind": "proposal", "x": 1}
    with pytest.raises(RuntimeError, match="provider failed"):
        exact_request("gpt", payload, output_tokens=1000, budget=budget(search_env),
                      call=lambda: (_ for _ in ()).throw(RuntimeError("provider failed")))
    with pytest.raises(RuntimeError, match="explicit retry"):
        exact_request("gpt", payload, output_tokens=1000, budget=budget(search_env),
                      call=lambda: {"answer": "no"})
    cache, = (search_env / "search_cache/gpt").glob("*.json")
    request_id = json.loads(cache.read_text())["request_id"]
    authorize_search_retry("gpt", request_id, authorized_by="alex", reason="transient outage")
    result = exact_request("gpt", payload, output_tokens=1000, budget=budget(search_env),
                           call=lambda: {"answer": "yes"})
    assert result == {"answer": "yes"}


def test_budget_stops_before_call(search_env):
    calls = []
    limited = budget(search_env, requests=1)
    exact_request("gpt", {"x": 1}, output_tokens=1000, budget=limited,
                  call=lambda: calls.append(1) or {"ok": True})
    with pytest.raises(RuntimeError, match="request limit"):
        exact_request("gpt", {"x": 2}, output_tokens=1000, budget=limited,
                      call=lambda: calls.append(2) or {"ok": True})
    assert calls == [1]


def test_openai_usage_is_normalized_and_reported(search_env):
    class LM:
        model = "openai/gpt-test"
        history = []

    lm = LM()
    def invoke():
        lm.history.append({"usage": {"prompt_tokens": 12, "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2}},
            "response_model": "gpt-test"})
        return {"ok": True}
    exact_request("gpt", {"x": "usage"}, output_tokens=1000,
                  budget=budget(search_env), call=invoke, lm=lm)
    report = usage_report(search_env / "search_cache")
    assert report["attempts"] == report["requests_with_usage"] == 1
    assert report["by_model"]["openai/gpt-test"]["input_tokens"] == 12
    assert report["by_model"]["openai/gpt-test"]["reasoning_tokens"] == 2


def test_anthropic_input_total_includes_provider_cache_counters():
    usage = normalize_usage("anthropic", {"input_tokens": 10, "output_tokens": 4,
        "cache_creation_input_tokens": 6, "cache_read_input_tokens": 8}, model="claude-test")
    assert usage["input_tokens"] == 24
    assert usage["cached_input_tokens"] == 8


def test_failed_parse_still_persists_provider_usage(search_env):
    class LM:
        model = "openai/gpt-test"
        history = []
    lm = LM()
    def invoke():
        lm.history.append({"usage": {"prompt_tokens": 9, "completion_tokens": 2},
                           "response_model": "gpt-test"})
        raise ValueError("bad parse")
    with pytest.raises(ValueError, match="bad parse"):
        exact_request("gpt", {"x": "failed-usage"}, output_tokens=1000,
                      budget=budget(search_env), call=invoke, lm=lm)
    report = usage_report(search_env / "search_cache")
    assert report["requests_with_usage"] == 1


def test_budget_survives_restart(search_env):
    first = budget(search_env, requests=1)
    first.reserve(10, 100)
    reopened = budget(search_env, requests=1)
    with pytest.raises(RuntimeError, match="request limit"):
        reopened.reserve(10, 100)


def test_started_attempt_is_durable_before_provider_call(search_env):
    def invoke():
        report = usage_report(search_env / "search_cache")
        assert report["attempts"] == report["unknown_usage_attempts"] == 1
        assert report["reserved_output_tokens"] == 1000
        return {"ok": True}
    exact_request("gpt", {"crash": "boundary"}, output_tokens=1000,
                  budget=budget(search_env), call=invoke)


def test_litellm_anthropic_usage_is_not_lost_or_double_counted():
    normalized = normalize_usage("anthropic", {"prompt_tokens": 24, "completion_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 8, "cache_creation_tokens": 6}})
    assert normalized["input_tokens"] == 24
    assert normalized["output_tokens"] == 4
    assert normalized["cache_write_input_tokens"] == 6
