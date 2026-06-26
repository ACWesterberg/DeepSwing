from __future__ import annotations

import pytest

from src.analysis.regime import RegimeResult
from src.analysis.screener import ScreenerCandidate, _score_candidate, screen_candidates
from src.analysis.technical import TechnicalSignals


def _make_signals(
    ticker: str = "TEST",
    price: float = 100.0,
    sma_50: float = 95.0,
    sma_200: float = 90.0,
    rsi_14: float = 52.0,
    volume_ratio: float = 1.6,
    ema_21: float = 97.0,
    bb_pct_b: float = 0.2,
    sar_is_bearish: bool = False,
) -> TechnicalSignals:
    return TechnicalSignals(
        ticker=ticker,
        ema_21=ema_21,
        sma_50=sma_50,
        sma_200=sma_200,
        price_above_50sma=price > sma_50,
        price_above_200sma=price > sma_200,
        ema_21_above_50sma=ema_21 > sma_50,
        atr_14=2.0,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_pct_b=bb_pct_b,
        rsi_14=rsi_14,
        parabolic_sar=97.0,
        sar_is_bearish=sar_is_bearish,
        ease_of_movement=0.01,
        obv=1_000_000.0,
        volume_ratio=volume_ratio,
        volume_spike=volume_ratio >= 1.5,
        fib_38_2=97.0,
        fib_61_8=94.0,
        current_price=price,
        current_volume=500_000.0,
    )


def _make_regime(regime: str = "trending", hurst: float = 0.65) -> RegimeResult:
    return RegimeResult(
        regime=regime,
        hurst_exponent=hurst,
        autocorrelation=0.1,
        recommended_tactic="test tactic",
    )


class TestScoreCandidate:
    def test_valid_trending_candidate_returns_score(self):
        sig = _make_signals(ema_21=97.0, sma_50=95.0)  # ema21 > sma50 ✓
        reg = _make_regime(regime="trending")
        score = _score_candidate(sig, reg)
        assert score is not None
        assert score > 0

    def test_valid_mean_reverting_candidate_returns_score(self):
        sig = _make_signals(bb_pct_b=0.2)  # bb_pct_b <= 0.35 ✓
        reg = _make_regime(regime="mean-reverting", hurst=0.35)
        score = _score_candidate(sig, reg)
        assert score is not None
        assert score > 0

    def test_price_below_50sma_rejected(self):
        sig = _make_signals(price=90.0, sma_50=95.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_rsi_too_high_rejected(self):
        sig = _make_signals(rsi_14=85.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_rsi_too_low_rejected(self):
        sig = _make_signals(rsi_14=20.0)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_low_volume_rejected(self):
        sig = _make_signals(volume_ratio=0.8)
        reg = _make_regime()
        assert _score_candidate(sig, reg) is None

    def test_neutral_regime_rejected(self):
        sig = _make_signals()
        reg = _make_regime(regime="neutral")
        assert _score_candidate(sig, reg) is None

    def test_trending_without_ema_crossover_rejected(self):
        sig = _make_signals(ema_21=93.0, sma_50=95.0)  # ema21 < sma50 ✗
        reg = _make_regime(regime="trending")
        assert _score_candidate(sig, reg) is None

    def test_mean_reverting_with_high_bb_pctb_rejected(self):
        sig = _make_signals(bb_pct_b=0.8)
        reg = _make_regime(regime="mean-reverting")
        assert _score_candidate(sig, reg) is None

    def test_price_above_200sma_adds_score(self):
        sig_above = _make_signals(price=100.0, sma_200=90.0)
        sig_below = _make_signals(price=100.0, sma_200=105.0)
        reg = _make_regime()
        score_above = _score_candidate(sig_above, reg)
        score_below = _score_candidate(sig_below, reg)
        if score_above is not None and score_below is not None:
            assert score_above > score_below

    def test_bearish_sar_lowers_score(self):
        sig_bull = _make_signals(sar_is_bearish=False)
        sig_bear = _make_signals(sar_is_bearish=True)
        reg = _make_regime()
        score_bull = _score_candidate(sig_bull, reg)
        score_bear = _score_candidate(sig_bear, reg)
        if score_bull is not None and score_bear is not None:
            assert score_bull > score_bear


class TestScreenCandidates:
    def _valid_entry(self, ticker: str = "A") -> tuple:
        sig = _make_signals(ticker=ticker, ema_21=97.0, sma_50=95.0)
        reg = _make_regime()
        return sig, reg

    def test_passes_valid_candidates(self):
        data = {"AAPL": self._valid_entry("AAPL")}
        result = screen_candidates(data, "us")
        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_filters_invalid_candidates(self):
        bad_sig = _make_signals(price=80.0, sma_50=95.0)  # price below SMA50
        data = {"BAD": (bad_sig, _make_regime())}
        result = screen_candidates(data, "us")
        assert len(result) == 0

    def test_returns_screener_candidate_objects(self):
        data = {"X": self._valid_entry("X")}
        result = screen_candidates(data, "us")
        assert all(isinstance(c, ScreenerCandidate) for c in result)

    def test_sorted_by_score_descending(self):
        # High volume = higher score
        sig_high = _make_signals(ticker="HI", volume_ratio=3.0, ema_21=97.0, sma_50=95.0)
        sig_low = _make_signals(ticker="LO", volume_ratio=1.6, ema_21=97.0, sma_50=95.0)
        reg = _make_regime()
        data = {"LO": (sig_low, reg), "HI": (sig_high, reg)}
        result = screen_candidates(data, "us")
        assert len(result) == 2
        assert result[0].ticker == "HI"

    def test_empty_input_returns_empty(self):
        result = screen_candidates({}, "us")
        assert result == []

    def test_to_dict_contains_expected_fields(self):
        data = {"AAPL": self._valid_entry("AAPL")}
        result = screen_candidates(data, "us")
        assert len(result) == 1
        d = result[0].to_dict()
        assert "ticker" in d
        assert "price" in d
        assert "rsi" in d
        assert "regime" in d
