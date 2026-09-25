import json
from datetime import date

import pytest

from src.agent.provider_usage import optimizer_usage_report
from src.agent.cost_estimate import price_usage_report


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def counters():
    return {"provider": "openai", "model": "gpt-5", "input_tokens": 1000,
            "output_tokens": 100, "cached_input_tokens": 0}


def test_mixed_transports_price_each_attempt_and_count_failures(tmp_path):
    save(tmp_path / "search_cache/gpt/request.json", {"attempts": [
        {"status": "failed", "reserved_output_tokens": 16000, "actual_usage": counters()},
        {"status": "complete", "reserved_output_tokens": 16000, "actual_usage": counters()},
    ]})
    save(tmp_path / "evaluations/gpt/run/batch.json", {
        "batch_id": "batch-1", "status": "complete", "custom_ids": ["shared"],
        "requests": {"shared": {"reserved_output_tokens": 16000}},
        "results": {"shared": {"usage": counters(), "body": {"prompt": "SECRET_PROMPT_SENTINEL"}}},
    })
    report = optimizer_usage_report(tmp_path)
    assert report["attempts"] == 3
    assert report["reserved_output_tokens"] == 48000
    assert report["by_model"]["openai/gpt-5"]["input_tokens"] == 3000
    price = price_usage_report(report, today=date(2026, 9, 23))
    assert price["usd"] == pytest.approx(.00225 * 2.5)
    assert "SECRET_PROMPT_SENTINEL" not in json.dumps(report)
    assert optimizer_usage_report(tmp_path) == report


@pytest.mark.parametrize("status", ["submitting", "submission_uncertain", "in_progress", "partial_failure"])
def test_unresolved_batch_attempt_prevents_partial_price(tmp_path, status):
    save(tmp_path / "evaluations/gpt/run/batch.json", {
        "status": status, "custom_ids": ["a", "b"],
        "results": {"a": {"usage": counters()}}, "errors": {},
    })
    report = optimizer_usage_report(tmp_path)
    assert report["attempts"] == 2
    assert report["unknown_usage_attempts"] == 1
    assert report["unknown_reservations"] == 2
    assert not price_usage_report(report, today=date(2026, 9, 23))["available"]


def test_upload_failure_is_not_counted_as_inference(tmp_path):
    save(tmp_path / "evaluations/gpt/run/batch.json", {"status": "upload_failed", "custom_ids": ["a"]})
    assert optimizer_usage_report(tmp_path)["attempts"] == 0


def test_retry_usage_adds_to_original_attempt_instead_of_replacing_it(tmp_path):
    for relative in ("batch.json", "retries/0001/batch.json"):
        save(tmp_path / "evaluations/gpt/run" / relative, {
            "status": "complete", "custom_ids": ["same-id"],
            "requests": {"same-id": {"reserved_output_tokens": 16000}},
            "results": {"same-id": {"usage": counters()}},
        })
    report = optimizer_usage_report(tmp_path)
    assert report["attempts"] == 2
    assert report["reserved_output_tokens"] == 32000
    assert report["by_model"]["openai/gpt-5"]["input_tokens"] == 2000
