from __future__ import annotations

import src.agent.erl as erl

_TRADE = {
    "id": 1, "ticker": "AAPL", "market": "us",
    "entry_price": 100.0, "exit_price": 110.0,
    "pnl_pct": 0.10, "duration_days": 3, "rrr_achieved": 2.0, "stop_hit": False,
}


class TestErlContext:
    def test_prompt_includes_news_and_macro(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(erl, "_call_model", lambda track, prompt: captured.setdefault("p", prompt) or "Quality: 0")
        erl.run_erl(
            "claude", dict(_TRADE),
            technicals_str="RSI 55, above EMA20",
            regime_str="trending",
            news_str="War escalates in the Gulf; energy names spike",
            macro_str="Fed on hold, CPI hot at 4.1%",
        )
        assert "War escalates in the Gulf" in captured["p"]
        assert "Fed on hold" in captured["p"]

    def test_prompt_falls_back_when_context_absent(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(erl, "_call_model", lambda track, prompt: captured.setdefault("p", prompt) or "Quality: 0")
        erl.run_erl("claude", dict(_TRADE), technicals_str="RSI 55", regime_str="trending")
        assert "No news/sentiment captured at entry." in captured["p"]
        assert "No macro context captured at entry." in captured["p"]
