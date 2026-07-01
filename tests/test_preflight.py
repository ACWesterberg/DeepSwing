from __future__ import annotations

import openai

from config.settings import settings
from src.scheduler import preflight


class TestLogModelConfig:
    def test_runs_without_error(self, caplog):
        import logging
        with caplog.at_level(logging.INFO):
            preflight.log_model_config()
        assert "Model configuration" in caplog.text


class TestCheckModels:
    def test_skips_providers_without_keys(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        assert preflight.check_models() == {}

    def test_pings_configured_models_when_keys_present(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "k")
        monkeypatch.setattr(settings, "openai_api_key", "k")
        # Stubbed clients (conftest) don't raise → every ping "succeeds"
        results = preflight.check_models()
        assert results
        assert all(results.values())
        # Both providers represented
        assert any(k.startswith("anthropic/") for k in results)
        assert any(k.startswith("openai/") for k in results)

    def test_dedupes_repeated_model_ids(self, monkeypatch):
        # erl and prompt both = opus → one anthropic entry for that ID
        monkeypatch.setattr(settings, "anthropic_api_key", "k")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "claude_decision_model", "claude-x")
        monkeypatch.setattr(settings, "claude_erl_model", "claude-y")
        monkeypatch.setattr(settings, "claude_prompt_model", "claude-y")
        results = preflight.check_models()
        assert set(results) == {"anthropic/claude-x", "anthropic/claude-y"}

    def test_reports_failure_when_ping_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "anthropic_api_key", "")
        monkeypatch.setattr(settings, "openai_api_key", "k")

        def _boom(*a, **k):
            raise RuntimeError("bad model id")

        monkeypatch.setattr(openai, "OpenAI", _boom)
        results = preflight.check_models()
        assert results  # openai models were attempted
        assert not any(results.values())  # all failed
