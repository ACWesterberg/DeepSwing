"""One transport attempt per reservation, with no adapter fallback."""
from types import SimpleNamespace


def single_predict(program, lm, inputs):
    from dspy.adapters.chat_adapter import ChatAdapter
    adapter = ChatAdapter()
    lm.num_retries = 0
    lm.cache = False
    outputs = lm(messages=adapter.format(program.signature, program.demos, inputs))
    if len(outputs) != 1 or not isinstance(outputs[0], str):
        raise ValueError("Expected exactly one text completion")
    return SimpleNamespace(**adapter.parse(program.signature, outputs[0]))


def effective_output_tokens(track, requested):
    return max(requested, 16000) if track == "gpt" else requested
