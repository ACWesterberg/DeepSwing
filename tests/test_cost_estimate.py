from datetime import date

import pytest

from src.agent.cost_estimate import estimate_usage_cost, price_usage_report


def usage(**overrides):
    return {"provider": "openai", "model": "gpt-5", "input_tokens": 1000,
            "cached_input_tokens": 200, "output_tokens": 100, **overrides}


def test_prices_uncached_cached_and_output_tokens():
    result = estimate_usage_cost(usage(), today=date(2026, 9, 23))
    assert result["available"]
    assert result["usd"] == pytest.approx((800 * 1.25 + 200 * .125 + 100 * 10) / 1_000_000)


def test_batch_applies_documented_half_price():
    standard = estimate_usage_cost(usage(), today=date(2026, 9, 23))["usd"]
    batch = estimate_usage_cost(usage(), batch=True, today=date(2026, 9, 23))["usd"]
    assert batch == standard / 2


@pytest.mark.parametrize("override", [
    {"model": "gpt-unknown"}, {"provider": "anthropic"}, {"output_tokens": None},
])
def test_unknown_inputs_fail_closed(override):
    assert not estimate_usage_cost(usage(**override), today=date(2026, 9, 23))["available"]


def test_stale_snapshot_fails_closed():
    result = estimate_usage_cost(usage(), today=date(2026, 10, 24))
    assert not result["available"] and "requires review" in result["reason"]


def test_report_refuses_partial_dollar_total():
    report = {"unknown_usage_attempts": 1, "by_model": {"openai/gpt-5": {
        "input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 2,
    }}}
    assert not price_usage_report(report, today=date(2026, 9, 23))["available"]


@pytest.mark.parametrize("inputs,long_context", [(272000, False), (272001, True)])
def test_sol_context_threshold_and_cache_write_partition(inputs, long_context):
    value = usage(model="gpt-5.6-sol", input_tokens=inputs, output_tokens=1000,
                  cached_input_tokens=10000, cache_write_input_tokens=20000)
    result = estimate_usage_cost(value, today=date(2026, 9, 23))
    expected = ((inputs - 30000) * 4 + 10000 * .4 + 20000 * 5) * (2 if long_context else 1)
    expected += 1000 * 20 * (1.5 if long_context else 1)
    assert result["usd"] == pytest.approx(expected / 1000000)
    assert result["long_context"] == long_context
    batch = estimate_usage_cost(value, batch=True, today=date(2026, 9, 23))
    assert batch["usd"] == result["usd"] / 2


def test_missing_cache_write_counter_is_unknown_not_zero():
    assert not estimate_usage_cost(usage(model="gpt-5.6-sol"), today=date(2026, 9, 23))["available"]


def test_impossible_cache_partition_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        estimate_usage_cost(usage(model="gpt-5.6-sol", cache_write_input_tokens=900), today=date(2026, 9, 23))


def test_long_context_pricing_is_not_applied_to_sum_of_short_requests():
    short = usage(model="gpt-5.6-sol", input_tokens=200000, cached_input_tokens=0, cache_write_input_tokens=0)
    report = {"attempt_details": [{"usage": short, "transport": "standard"} for _ in range(2)]}
    result = price_usage_report(report, today=date(2026, 9, 23))
    assert result["usd"] == pytest.approx(2 * estimate_usage_cost(short, today=date(2026, 9, 23))["usd"])
    legacy = {"by_model": {"openai/gpt-5.6-sol": {**short, "input_tokens": 400000}}}
    assert not price_usage_report(legacy, today=date(2026, 9, 23))["available"]
