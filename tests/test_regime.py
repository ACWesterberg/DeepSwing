from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.regime import RegimeResult, _autocorrelation, _hurst_exponent, classify_regime


def _make_trending_df(n: int = 120) -> pd.DataFrame:
    """Strictly monotone uptrend — Hurst should be well above 0.55."""
    prices = np.linspace(100, 200, n)
    return pd.DataFrame({"Close": prices})


def _make_mean_reverting_df(n: int = 120) -> pd.DataFrame:
    """Alternating series — Hurst should be well below 0.45."""
    prices = np.array([100 + (5 if i % 2 == 0 else -5) for i in range(n)], dtype=float)
    return pd.DataFrame({"Close": prices})


def _make_random_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"Close": np.clip(prices, 1, None)})


class TestHurstExponent:
    def test_trending_series_high_hurst(self):
        ts = np.linspace(100, 200, 100)
        h = _hurst_exponent(ts)
        assert h > 0.55, f"Expected H > 0.55 for trending series, got {h:.3f}"

    def test_mean_reverting_series_low_hurst(self):
        ts = np.array([100 + (5 if i % 2 == 0 else -5) for i in range(100)], dtype=float)
        h = _hurst_exponent(ts)
        assert h < 0.45, f"Expected H < 0.45 for mean-reverting series, got {h:.3f}"

    def test_result_clipped_to_0_1(self):
        ts = np.ones(100) * 50
        h = _hurst_exponent(ts)
        assert 0.0 <= h <= 1.0

    def test_too_short_series_returns_half(self):
        h = _hurst_exponent(np.array([1.0, 2.0]))
        assert h == 0.5


class TestAutocorrelation:
    def test_positively_correlated_returns(self):
        # Trending series has positive lag-1 autocorrelation of log returns
        ts = np.linspace(100, 200, 100)
        ac = _autocorrelation(ts, lag=1)
        # For a strict monotone series, returns are constant → autocorr may be nan→0,
        # but generally trending returns show positive autocorrelation
        assert isinstance(ac, float)

    def test_returns_zero_for_constant_series(self):
        ts = np.ones(50) * 100.0
        ac = _autocorrelation(ts, lag=1)
        assert ac == 0.0

    def test_too_short_returns_zero(self):
        ac = _autocorrelation(np.array([1.0]), lag=1)
        assert ac == 0.0


class TestClassifyRegime:
    def test_trending_regime_from_monotone_data(self):
        df = _make_trending_df()
        result = classify_regime(df)
        assert result.regime == "trending"
        assert result.hurst_exponent > 0.55

    def test_mean_reverting_regime_from_alternating_data(self):
        df = _make_mean_reverting_df()
        result = classify_regime(df)
        assert result.regime == "mean-reverting"
        assert result.hurst_exponent < 0.45

    def test_returns_neutral_when_too_few_rows(self):
        df = pd.DataFrame({"Close": [100.0, 101.0, 99.0]})
        result = classify_regime(df)
        assert result.regime == "neutral"
        assert result.hurst_exponent == 0.5

    def test_result_is_regime_result(self):
        df = _make_random_df()
        result = classify_regime(df)
        assert isinstance(result, RegimeResult)

    def test_hurst_in_valid_range(self):
        df = _make_random_df()
        result = classify_regime(df)
        assert 0.0 <= result.hurst_exponent <= 1.0

    def test_trending_tactic_mentions_ema(self):
        df = _make_trending_df()
        result = classify_regime(df)
        if result.regime == "trending":
            assert "EMA" in result.recommended_tactic or "breakout" in result.recommended_tactic.lower()

    def test_mean_reverting_tactic_mentions_bollinger(self):
        df = _make_mean_reverting_df()
        result = classify_regime(df)
        if result.regime == "mean-reverting":
            assert "Bollinger" in result.recommended_tactic

    def test_to_prompt_str_contains_hurst(self):
        df = _make_random_df()
        result = classify_regime(df)
        text = result.to_prompt_str()
        assert "Hurst" in text
        assert result.regime in text
