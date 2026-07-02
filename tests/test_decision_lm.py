from __future__ import annotations

import dspy

from src.agent.decision import build_lm


class TestBuildLm:
    def test_gpt_uses_reasoning_model_params(self):
        dspy.LM.reset_mock()
        build_lm("gpt", "gpt-5", "key")
        _, kwargs = dspy.LM.call_args
        assert kwargs["model"] == "openai/gpt-5"
        assert kwargs["temperature"] == 1.0
        assert kwargs["max_tokens"] >= 16000

    def test_gpt_enforces_max_tokens_floor(self):
        dspy.LM.reset_mock()
        build_lm("gpt", "gpt-5", "key", max_tokens=4096)  # below the 16000 floor
        _, kwargs = dspy.LM.call_args
        assert kwargs["max_tokens"] == 16000

    def test_claude_keeps_requested_max_tokens_and_no_temperature(self):
        dspy.LM.reset_mock()
        build_lm("claude", "claude-sonnet-5", "key", max_tokens=1024)
        _, kwargs = dspy.LM.call_args
        assert kwargs["model"] == "anthropic/claude-sonnet-5"
        assert kwargs["max_tokens"] == 1024
        assert "temperature" not in kwargs
