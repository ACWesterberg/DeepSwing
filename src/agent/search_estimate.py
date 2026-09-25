"""Local advisory estimates for a frozen, already-split optimizer corpus."""
import json

from config.settings import settings
from src.agent.cost_estimate import estimate_usage_cost, PRICES


def estimate_requests(requests, *, encoder_for_model, today=None):
    rows = []
    for request in requests:
        model = request["model"]
        messages = request["messages"]
        allowance = request.get("instruction_allowance", 0) * 4
        wire = json.dumps(messages, ensure_ascii=False)
        row = {"role": request["role"], "model": model, "batch": request["batch"],
               "context_bytes": len(wire.encode()), "candidate_allowance_bytes": allowance,
               "output_max": request["output_tokens"]}
        try:
            if request["provider"] != "openai":
                raise ValueError("No verified offline tokenizer for this provider")
            encoding = encoder_for_model(model)
            # Server framing is not fully specified. This is a local advisory
            # count, with an explicit allowance, never a billed-token guarantee.
            count = sum(len(encoding.encode(m["content"], disallowed_special=())) for m in messages)
            count += 32 * len(messages)
            upper = count + allowance
            low = estimate_usage_cost({"provider": "openai", "model": model, "input_tokens": count,
                "output_tokens": 0, "cached_input_tokens": 0, "cache_write_input_tokens": 0},
                batch=request["batch"], today=today)
            high = estimate_usage_cost({"provider": "openai", "model": model, "input_tokens": upper,
                "output_tokens": request["output_tokens"], "cached_input_tokens": 0,
                "cache_write_input_tokens": upper if "cache_write" in PRICES.get(model, {}) else 0},
                batch=request["batch"], today=today)
            row.update(input_low=count, input_high=upper, encoding=encoding.name)
            row["pricing_available"] = low["available"] and high["available"]
            if row["pricing_available"]:
                row.update(usd_low=low["usd"], usd_high=high["usd"])
            else:
                row["unavailable_reason"] = low.get("reason") or high.get("reason")
        except Exception as exc:
            row.update(pricing_available=False, unavailable_reason=str(exc))
        rows.append(row)
    available = bool(rows) and all(row["pricing_available"] for row in rows)
    result = {"requests": len(rows), "rows": rows, "pricing_available": available,
              "reserved_output_tokens": sum(row["output_max"] for row in rows),
              "scope": "Whole-plan estimate before exact-cache reuse; no provider calls or evidence reservation",
              "assumptions": "Approximate framing; no cache-read savings; upper includes maximum output and cache writes. Not a billing cap."}
    if available:
        result.update(usd_low=sum(row["usd_low"] for row in rows), usd_high=sum(row["usd_high"] for row in rows))
    return result


def search_requests(datasets, *, track, incumbent, candidate=None, batch=False):
    import dspy
    from dspy.adapters.chat_adapter import ChatAdapter
    from src.agent.decision import TradeDecision
    from src.scheduler.optimizer import InstructionProposal
    from src.agent.bounded_search import proposal_fields, select_screen_examples
    from src.agent.replay import DECISION_INPUTS
    from src.agent.single_request import effective_output_tokens
    adapter = ChatAdapter()
    provider = "openai" if track == "gpt" else "anthropic"
    model = settings.gpt_decision_model if track == "gpt" else settings.claude_decision_model
    proposer_model = settings.gpt_prompt_model if track == "gpt" else settings.claude_prompt_model
    output = effective_output_tokens(track, settings.bounded_search_output_tokens_per_request)
    def request(role, model, program, inputs, is_batch, allowance=0):
        return {"role": role, "provider": provider, "model": model, "output_tokens": output,
                "messages": adapter.format(program.signature, program.demos, inputs),
                "batch": is_batch, "instruction_allowance": allowance}
    fields = proposal_fields(datasets["train"] + datasets["validation"], incumbent.signature.instructions)
    rows = [request("proposal", proposer_model, dspy.Predict(InstructionProposal), fields, False)]
    unknown = candidate is None
    candidate = candidate or dspy.Predict(TradeDecision.with_instructions(" "))
    for example in select_screen_examples(datasets["test"], settings.bounded_search_examples):
        inputs = {k: example[k] for k in DECISION_INPUTS}
        rows.append(request("incumbent", model, incumbent, inputs, batch and track == "gpt"))
        rows.append(request("candidate", model, candidate, inputs, batch and track == "gpt",
                            settings.bounded_search_instruction_max_chars if unknown else 0))
    return rows
