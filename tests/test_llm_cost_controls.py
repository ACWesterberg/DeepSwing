from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from config.settings import settings
from src.agent import news_analyzer, openai_client
from src.agent.news_analyzer import analyze_news


class _Resp:
    def __init__(self, content: str):
        message = MagicMock()
        message.content = content
        choice = MagicMock()
        choice.message = message
        self.choices = [choice]


@pytest.fixture
def captured_calls(monkeypatch):
    """Record every kwargs dict handed to chat.completions.create."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return _Resp("ok")

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    monkeypatch.setattr(sys.modules["openai"], "OpenAI", lambda **_: client)
    openai_client._NO_REASONING_SUPPORT.clear()
    return calls


class TestLightCompletion:
    def test_sends_configured_reasoning_effort(self, captured_calls, monkeypatch):
        monkeypatch.setattr(settings, "gpt_light_reasoning_effort", "low")
        assert openai_client.light_completion("gpt-5-mini", "hi", 500) == "ok"
        assert captured_calls[0]["reasoning_effort"] == "low"
        assert captured_calls[0]["max_completion_tokens"] == 500

    def test_empty_effort_sends_no_parameter(self, captured_calls, monkeypatch):
        """A non-reasoning model rejects the field outright — "" is the escape."""
        monkeypatch.setattr(settings, "gpt_light_reasoning_effort", "")
        openai_client.light_completion("gpt-4o-mini", "hi", 500)
        assert "reasoning_effort" not in captured_calls[0]

    def test_falls_back_once_when_model_rejects_the_parameter(self, monkeypatch):
        monkeypatch.setattr(settings, "gpt_light_reasoning_effort", "low")
        calls: list[dict] = []

        def create(**kwargs):
            calls.append(kwargs)
            if "reasoning_effort" in kwargs:
                raise ValueError("Unsupported parameter: 'reasoning_effort'")
            return _Resp("ok")

        client = MagicMock()
        client.chat.completions.create.side_effect = create
        monkeypatch.setattr(sys.modules["openai"], "OpenAI", lambda **_: client)
        openai_client._NO_REASONING_SUPPORT.clear()

        assert openai_client.light_completion("legacy-model", "hi", 500) == "ok"
        assert len(calls) == 2 and "reasoning_effort" not in calls[1]

        # The rejection is remembered, so the next call doesn't buy the retry.
        assert openai_client.light_completion("legacy-model", "hi", 500) == "ok"
        assert len(calls) == 3 and "reasoning_effort" not in calls[2]

    def test_other_errors_propagate_without_a_second_call(self, monkeypatch):
        """A rate limit must not be retried here — that doubles the spend."""
        monkeypatch.setattr(settings, "gpt_light_reasoning_effort", "low")
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("429 rate limit")
        monkeypatch.setattr(sys.modules["openai"], "OpenAI", lambda **_: client)
        openai_client._NO_REASONING_SUPPORT.clear()

        with pytest.raises(RuntimeError):
            openai_client.light_completion("gpt-5-mini", "hi", 500)
        assert client.chat.completions.create.call_count == 1


@pytest.fixture(autouse=True)
def _clear_summary_cache():
    news_analyzer._SUMMARY_CACHE.clear()
    yield
    news_analyzer._SUMMARY_CACHE.clear()


def _articles(headline: str) -> list[dict]:
    return [{"headline": headline, "source": "Test", "published_at": "2026-01-01"}]


class TestNewsSummaryCache:
    def test_identical_articles_are_analyzed_once(self, monkeypatch):
        """The article fetch is TTL-cached, so consecutive scans see the same
        headlines — paying for the same summary each time was pure waste."""
        monkeypatch.setattr(settings, "news_refresh_interval_minutes", 60)
        calls: list[str] = []
        monkeypatch.setattr(
            news_analyzer, "light_completion",
            lambda model, prompt, max_completion_tokens: calls.append(prompt) or "summary",
        )

        first = analyze_news("AAPL", "us", 100.0, "brief", _articles("AAPL earnings beat"))
        # A later scan: same news, price has drifted, technicals brief differs.
        second = analyze_news("AAPL", "us", 101.5, "other brief", _articles("AAPL earnings beat"))

        assert first == second == "summary"
        assert len(calls) == 1

    def test_new_headline_forces_a_fresh_analysis(self, monkeypatch):
        monkeypatch.setattr(settings, "news_refresh_interval_minutes", 60)
        calls: list[str] = []
        monkeypatch.setattr(
            news_analyzer, "light_completion",
            lambda model, prompt, max_completion_tokens: calls.append(prompt) or "summary",
        )

        analyze_news("AAPL", "us", 100.0, "brief", _articles("AAPL earnings beat"))
        analyze_news("AAPL", "us", 100.0, "brief", _articles("AAPL downgrade"))
        assert len(calls) == 2

    def test_exit_review_bypasses_the_cache(self, monkeypatch):
        """_maybe_news_exit forced a fresh fetch because freshness decides an
        exit; it must not read back a summary framed at entry."""
        monkeypatch.setattr(settings, "news_refresh_interval_minutes", 60)
        calls: list[str] = []
        monkeypatch.setattr(
            news_analyzer, "light_completion",
            lambda model, prompt, max_completion_tokens: calls.append(prompt) or "summary",
        )

        analyze_news("AAPL", "us", 100.0, "brief", _articles("AAPL earnings beat"))
        analyze_news(
            "AAPL", "us", 108.0, "held", _articles("AAPL earnings beat"), use_cache=False
        )
        assert len(calls) == 2

    def test_cache_is_bounded(self, monkeypatch):
        monkeypatch.setattr(settings, "news_refresh_interval_minutes", 60)
        monkeypatch.setattr(
            news_analyzer, "light_completion",
            lambda model, prompt, max_completion_tokens: "summary",
        )
        for i in range(news_analyzer._SUMMARY_CACHE_MAX + 25):
            analyze_news(f"T{i}", "us", 100.0, "brief", _articles(f"earnings {i}"))
        assert len(news_analyzer._SUMMARY_CACHE) <= news_analyzer._SUMMARY_CACHE_MAX
