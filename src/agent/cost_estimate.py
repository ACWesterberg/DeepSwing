"""Dated advisory pricing for provider-reported optimizer usage."""
from __future__ import annotations

from datetime import date


PRICE_DATE = date(2026, 9, 23)
PRICE_VALID_UNTIL = date(2026, 10, 23)
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5"
BATCH_SOURCE = "https://developers.openai.com/api/docs/guides/batch"

# USD per million tokens at the standard service tier. Keep this intentionally
# narrow: an unknown alias or a stale snapshot makes the whole estimate
# unavailable instead of silently borrowing another model's rate.
PRICES = {
    "gpt-5": {"input": 1.25, "cached_input": .125, "output": 10.0},
    "gpt-5-2025-08-07": {"input": 1.25, "cached_input": .125, "output": 10.0},
    "gpt-5.6-sol": {"input": 4.0, "cached_input": .4, "cache_write": 5.0,
                    "output": 20.0, "long_context_threshold": 272000},
    "claude-sonnet-5": {"input": 2., "cached_input": .2, "cache_write": 2.5, "output": 10., "cache_write_1h": 4.},
    "claude-opus-4-8": {"input": 5., "cached_input": .5, "cache_write": 6.25, "output": 25., "cache_write_1h": 10.},
}


def estimate_usage_cost(usage: dict, *, batch: bool = False, today: date | None = None) -> dict:
    """Price normalized provider counters; this is advisory, never a budget cap."""
    today = today or date.today()
    if not PRICE_DATE <= today <= PRICE_VALID_UNTIL:
        return {"available": False,
                "reason": f"Pricing snapshot dated {PRICE_DATE.isoformat()} requires review"}
    if usage.get("provider") not in ("openai", "anthropic"):
        return {"available": False, "reason": "No verified pricing snapshot for this provider"}
    model = usage.get("model")
    rates = PRICES.get(model)
    if rates is None:
        return {"available": False, "reason": f"No verified price for model {model!r}"}
    if (model.startswith("claude-") and usage["provider"] != "anthropic") or (model.startswith("gpt-") and usage["provider"] != "openai"):
        return {"available": False, "reason": "Model and provider do not match"}
    if usage.get("requests", 1) != 1:
        return {"available": False, "reason": "Pricing requires individual request usage"}
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    cached = usage.get("cached_input_tokens")
    writes = usage.get("cache_write_input_tokens")
    if inputs is None or outputs is None:
        return {"available": False, "reason": "Provider input/output usage is incomplete"}
    if "cache_write" in rates and (writes is None or cached is None):
        return {"available": False, "reason": "Cache read/write counters are required for this model"}
    cached = 0 if cached is None else cached
    writes = 0 if writes is None else writes
    if not all(type(value) is int and value >= 0 for value in (inputs, outputs, cached, writes)):
        raise ValueError("Usage counters must be nonnegative integers")
    if cached + writes > inputs:
        raise ValueError("Cache reads and writes cannot exceed total input")
    write_cost = writes * rates.get("cache_write", rates["input"])
    if "cache_write_1h" in rates and writes:
        five, hour = usage.get("cache_write_5m_tokens"), usage.get("cache_write_1h_tokens")
        if five is None or hour is None:
            return {"available": False, "reason": "Anthropic cache-write durations are unavailable"}
        if any(type(v) is not int or v < 0 for v in (five, hour)) or five + hour != writes:
            raise ValueError("Invalid Anthropic cache-write duration counters")
        write_cost = five * rates["cache_write"] + hour * rates["cache_write_1h"]
    long_context = inputs > rates.get("long_context_threshold", float("inf"))
    input_multiplier = 2 if long_context else 1
    output_multiplier = 1.5 if long_context else 1
    multiplier = .5 if batch else 1.0
    usd = multiplier * (((inputs - cached - writes) * rates["input"]
                         + cached * rates["cached_input"]
                         + write_cost) * input_multiplier
                         + outputs * rates["output"] * output_multiplier) / 1_000_000
    return {"available": True, "usd": usd, "model": model, "batch": batch,
            "long_context": long_context,
            "price_date": PRICE_DATE.isoformat(), "valid_until": PRICE_VALID_UNTIL.isoformat(),
            "source": ("https://platform.claude.com/docs/en/about-claude/pricing" if usage["provider"] == "anthropic" else
                       "https://developers.openai.com/api/docs/models/gpt-5.6-sol" if model == "gpt-5.6-sol" else PRICE_SOURCE),
            "batch_source": BATCH_SOURCE if usage["provider"] == "openai" else "https://platform.claude.com/docs/en/about-claude/pricing"}


def price_usage_report(report: dict, *, batch: bool = False, today: date | None = None) -> dict:
    """Price a usage report only when every model total has a valid quote."""
    if "attempt_details" in report:
        total = 0.0
        for attempt in report["attempt_details"]:
            usage = attempt.get("usage")
            if not usage or usage.get("requests", 1) != 1:
                return {"available": False, "reason": "Unknown or aggregated request usage prevents a complete quote"}
            estimate = estimate_usage_cost(usage, batch=attempt["transport"] == "openai_batch", today=today)
            if not estimate["available"]:
                return estimate
            total += estimate["usd"]
        return {"available": True, "usd": total, "price_date": PRICE_DATE.isoformat(),
                "scope": "Per-attempt standard/Batch pricing; excludes taxes and account-specific charges"}
    rows, total = {}, 0.0
    for key, counters in report.get("by_model", {}).items():
        provider, model = key.split("/", 1)
        if "long_context_threshold" in PRICES.get(model, {}):
            return {"available": False, "reason": "Aggregate totals cannot determine per-request context pricing"}
        estimate = estimate_usage_cost({**counters, "provider": provider, "model": model},
                                       batch=batch, today=today)
        if not estimate["available"]:
            return {"available": False, "reason": estimate["reason"]}
        rows[key] = estimate
        total += estimate["usd"]
    if report.get("unknown_usage_attempts"):
        return {"available": False, "reason": "At least one attempt has unknown provider usage"}
    return {"available": True, "usd": total, "models": rows,
            "price_date": PRICE_DATE.isoformat(), "batch": batch}
