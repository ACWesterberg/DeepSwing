from __future__ import annotations

import pytest

from config.settings import settings
from src.analysis.regime import RegimeResult
from src.analysis.screener import ScreenerCandidate
from src.analysis.technical import TechnicalSignals
from src.analysis import triage
from src.analysis.triage import triage_candidates


def _make_signals(ticker: str = "TEST", price: float = 100.0) -> TechnicalSignals:
    return TechnicalSignals(
        ticker=ticker,
        ema_21=97.0,
        sma_50=95.0,
        sma_200=90.0,
        price_above_50sma=True,
        price_above_200sma=True,
        ema_21_above_50sma=True,
        atr_14=2.0,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_pct_b=0.2,
        rsi_14=52.0,
        parabolic_sar=97.0,
        sar_is_bearish=False,
        ease_of_movement=0.01,
        obv=1_000_000.0,
        volume_ratio=1.6,
        volume_spike=True,
        fib_38_2=97.0,
        fib_61_8=94.0,
        current_price=price,
        current_volume=500_000.0,
    )


def _make_regime() -> RegimeResult:
    return RegimeResult(
        regime="trending",
        hurst_exponent=0.65,
        autocorrelation=0.1,
        recommended_tactic="test tactic",
    )


def _candidates(tickers: list[str]) -> list[ScreenerCandidate]:
    return [ScreenerCandidate(t, "us", _make_signals(t), _make_regime()) for t in tickers]


@pytest.fixture(autouse=True)
def _triage_settings(monkeypatch):
    monkeypatch.setattr(settings, "triage_enabled", True)
    monkeypatch.setattr(settings, "triage_keep_top", 2)


class TestTriage:
    def test_disabled_returns_all(self, monkeypatch):
        monkeypatch.setattr(settings, "triage_enabled", False)
        cands = _candidates(["A", "B", "C", "D"])
        assert triage_candidates(cands, "us") == cands

    def test_keep_zero_returns_all(self, monkeypatch):
        monkeypatch.setattr(settings, "triage_keep_top", 0)
        cands = _candidates(["A", "B", "C"])
        assert triage_candidates(cands, "us") == cands

    def test_small_set_skips_llm_call(self, monkeypatch):
        def _boom(prompt):
            raise AssertionError("triage model should not be called")

        monkeypatch.setattr(triage, "_call_triage_model", _boom)
        cands = _candidates(["A", "B"])
        assert triage_candidates(cands, "us") == cands

    def test_keeps_chosen_tickers_in_screener_order(self, monkeypatch):
        monkeypatch.setattr(triage, "_call_triage_model", lambda p: '["D", "B"]')
        cands = _candidates(["A", "B", "C", "D"])
        kept = triage_candidates(cands, "us")
        assert [c.ticker for c in kept] == ["B", "D"]

    def test_parses_json_embedded_in_prose(self, monkeypatch):
        monkeypatch.setattr(
            triage, "_call_triage_model",
            lambda p: 'The strongest setups are:\n["C", "A"]\nbased on trend.',
        )
        cands = _candidates(["A", "B", "C"])
        assert [c.ticker for c in triage_candidates(cands, "us")] == ["A", "C"]

    def test_api_error_falls_back_to_screener_top_k(self, monkeypatch):
        def _boom(prompt):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(triage, "_call_triage_model", _boom)
        cands = _candidates(["A", "B", "C", "D"])
        assert [c.ticker for c in triage_candidates(cands, "us")] == ["A", "B"]

    def test_garbage_reply_falls_back_to_screener_top_k(self, monkeypatch):
        monkeypatch.setattr(triage, "_call_triage_model", lambda p: "no array here")
        cands = _candidates(["A", "B", "C"])
        assert [c.ticker for c in triage_candidates(cands, "us")] == ["A", "B"]

    def test_unknown_tickers_in_reply_fall_back(self, monkeypatch):
        monkeypatch.setattr(triage, "_call_triage_model", lambda p: '["ZZZ", "QQQ"]')
        cands = _candidates(["A", "B", "C"])
        assert [c.ticker for c in triage_candidates(cands, "us")] == ["A", "B"]

    def test_sides_reach_the_prompt(self, monkeypatch):
        seen = {}

        def _capture(prompt):
            seen["prompt"] = prompt
            return '["A", "B"]'

        monkeypatch.setattr(triage, "_call_triage_model", _capture)
        cands = _candidates(["A", "B", "C"])
        triage_candidates(cands, "US options", sides={"A": "call", "B": "put", "C": "put"})
        assert "direction: call" in seen["prompt"]
        assert "direction: put" in seen["prompt"]
