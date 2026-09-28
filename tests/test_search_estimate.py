from datetime import date

import pytest

from src.agent.search_estimate import estimate_requests


class Encoder:
    name = "test-byte-encoder"

    def encode(self, text, **kwargs):
        return list(text.encode())


def request(**overrides):
    return dict(role="candidate", provider="openai", model="gpt-5",
                messages=[{"role": "user", "content": "hello"}],
                output_tokens=100, batch=False, instruction_allowance=10, **overrides)


def estimate(rows, **kwargs):
    return estimate_requests(rows, encoder_for_model=lambda model: Encoder(),
                             today=date(2026, 9, 25), **kwargs)


def test_size_allowance_and_output_are_included():
    result = estimate([request()])
    row = result["rows"][0]
    assert row["input_low"] == 37
    assert row["input_high"] == 77
    assert row["candidate_allowance_bytes"] == 40
    assert result["reserved_output_tokens"] == 100
    assert result["usd_high"] == pytest.approx((77 * 1.25 + 100 * 10) / 1e6)


def test_batch_prices_only_selected_requests():
    ordinary = request()
    batched = {**ordinary, "batch": True}
    one = estimate([ordinary])["usd_high"]
    assert estimate([ordinary, batched])["usd_high"] == pytest.approx(one * 1.5)


@pytest.mark.parametrize("change", [{"provider": "anthropic"}, {"model": "unknown"}])
def test_unknown_request_prevents_partial_total(change):
    result = estimate([request(), {**request(), **change}])
    assert not result["pricing_available"]
    assert "usd_high" not in result
    assert result["rows"][1]["context_bytes"] > 0


def test_missing_local_tokenizer_does_not_fail_the_report():
    def missing(model):
        raise RuntimeError("Offline tokenizer unavailable")
    result = estimate_requests([request()], encoder_for_model=missing)
    assert not result["pricing_available"]
    assert "Offline" in result["rows"][0]["unavailable_reason"]


def test_expired_prices_and_empty_plans_are_not_quotes():
    result = estimate_requests([request()], encoder_for_model=lambda model: Encoder(), today=date(2027, 1, 1))
    assert not result["pricing_available"]
    assert not estimate([])["pricing_available"]
