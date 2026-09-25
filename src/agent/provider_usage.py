"""Normalize provider usage counters and report durable search consumption."""
from __future__ import annotations

import json
from pathlib import Path


def normalize_usage(provider: str, raw: dict | None, *, model: str | None = None) -> dict | None:
    """Return comparable counters without converting missing values to zero."""
    if not raw:
        return None
    if provider == "openai" or "prompt_tokens" in raw:
        details_in = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
        details_out = raw.get("completion_tokens_details") or raw.get("output_tokens_details") or {}
        inputs = raw.get("prompt_tokens", raw.get("input_tokens"))
        outputs = raw.get("completion_tokens", raw.get("output_tokens"))
        cached = details_in.get("cached_tokens")
        cache_write = details_in.get("cache_write_tokens", details_in.get("cache_creation_tokens"))
        reasoning = details_out.get("reasoning_tokens")
    elif provider == "anthropic":
        cache_write = raw.get("cache_creation_input_tokens")
        cached = raw.get("cache_read_input_tokens")
        inputs = raw.get("input_tokens")
        if inputs is not None:
            inputs += (cache_write or 0) + (cached or 0)
        outputs = raw.get("output_tokens")
        reasoning = None
    else:
        return None
    creation = raw.get("cache_creation") or (raw.get("prompt_tokens_details") or {}).get("cache_creation_token_details") or {}
    return {
        "provider": provider, "model": model,
        "input_tokens": inputs, "output_tokens": outputs,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "cache_write_5m_tokens": creation.get("ephemeral_5m_input_tokens"),
        "cache_write_1h_tokens": creation.get("ephemeral_1h_input_tokens"),
        "reasoning_tokens": reasoning, "raw": raw,
    }


def usage_from_lm_history(lm, start: int) -> dict | None:
    """Aggregate provider counters added to one DSPy LM after ``start``."""
    provider = str(getattr(lm, "model", "")).split("/", 1)[0]
    rows = []
    for entry in getattr(lm, "history", [])[start:]:
        normalized = normalize_usage(
            provider, entry.get("usage"), model=entry.get("response_model") or entry.get("model")
        )
        if normalized:
            rows.append(normalized)
    if not rows:
        return None
    fields = ("input_tokens", "output_tokens", "cached_input_tokens",
              "cache_write_input_tokens", "reasoning_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens")
    result = {"provider": provider, "model": rows[-1].get("model"), "requests": len(rows)}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[field] = sum(values) if len(values) == len(rows) else None
    result["raw"] = [row["raw"] for row in rows]
    return result


def usage_report(root: Path) -> dict:
    """Summarize each provider attempt once; unknown usage remains explicit."""
    rows = []
    for path in sorted(root.glob("*/*.json")):
        record = json.loads(path.read_text())
        attempts = record.get("attempts")
        if attempts is None or (not attempts and record.get("status") == "started"):
            attempts = [{"reserved_output_tokens": record.get("reserved_output_tokens", 0),
                         "actual_usage": record.get("actual_usage")}]
        for attempt in attempts:
            rows.append({"source": str(path), "transport": "standard", **attempt})
    return summarize_attempts(rows)


def summarize_attempts(attempts: list[dict]) -> dict:
    report = {"attempts": 0, "requests_with_usage": 0, "unknown_usage_attempts": 0,
              "reserved_output_tokens": 0, "unknown_reservations": 0,
              "by_model": {}, "attempt_details": []}
    for attempt in attempts:
        report["attempts"] += 1
        reservation = attempt.get("reserved_output_tokens")
        if reservation is None:
            report["unknown_reservations"] += 1
        elif type(reservation) is not int or reservation < 0:
            raise ValueError("Invalid output token reservation")
        else:
            report["reserved_output_tokens"] += reservation
        usage = attempt.get("actual_usage")
        # Do not include raw prompts or provider bodies in the public report.
        report["attempt_details"].append({
            "source": attempt.get("source"), "status": attempt.get("status"),
            "transport": attempt.get("transport", "standard"),
            "reserved_output_tokens": reservation,
            "usage": {k: v for k, v in usage.items() if k != "raw"} if isinstance(usage, dict) else None,
        })
        if not isinstance(usage, dict) or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
            report["unknown_usage_attempts"] += 1
            continue
        report["requests_with_usage"] += 1
        key = f"{usage.get('provider') or 'unknown'}/{usage.get('model') or 'unknown'}"
        totals = report["by_model"].setdefault(key, {
            "requests": 0, "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "cache_write_input_tokens": 0,
            "reasoning_tokens": 0,
        })
        totals["requests"] += usage.get("requests", 1)
        for field in ("input_tokens", "output_tokens", "cached_input_tokens",
                      "cache_write_input_tokens", "reasoning_tokens"):
            value = usage.get(field)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise ValueError(f"Invalid provider usage in {attempt.get('source')}")
                totals[field] += value
    return report


def optimizer_usage_report(compiled_root: Path) -> dict:
    """Count synchronous cache attempts and each submitted Batch job once."""
    synchronous = usage_report(compiled_root / "search_cache")
    attempts = [{**row, "actual_usage": row["usage"]} for row in synchronous["attempt_details"]]
    evaluation_root = compiled_root / "evaluations"
    paths = list(evaluation_root.glob("*/*/batch.json")) + list(evaluation_root.glob("*/*/retries/*/batch.json"))
    for path in sorted(paths):
        state = json.loads(path.read_text())
        # Upload alone is not inference. Submitting without an ID is ambiguous
        # and must be counted as unknown consumption, including process crashes.
        if state.get("status") in ("uploading", "upload_failed") and not state.get("batch_id"):
            continue
        for ident in dict.fromkeys(state["custom_ids"]):
            request = state.get("requests", {}).get(ident, {})
            result = state.get("results", {}).get(ident) or state.get("errors", {}).get(ident) or {}
            attempts.append({"source": f"{path}#{ident}", "transport": "openai_batch",
                "status": "complete" if ident in state.get("results", {}) else state.get("status"),
                "reserved_output_tokens": request.get("reserved_output_tokens"),
                "actual_usage": result.get("usage")})
    return summarize_attempts(attempts)
