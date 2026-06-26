from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.technical import TechnicalSignals, _last, compute_signals


def _make_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    prices = np.clip(prices, 1, None)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    high = prices * (1 + rng.uniform(0, 0.02, n))
    low = prices * (1 - rng.uniform(0, 0.02, n))
    return pd.DataFrame({
        "Open": prices,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    })


class TestLast:
    def test_returns_last_non_nan(self):
        s = pd.Series([1.0, 2.0, float("nan")])
        assert _last(s) is None

    def test_returns_float(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _last(s) == pytest.approx(3.0)

    def test_none_series(self):
        assert _last(None) is None


class TestComputeSignals:
    def test_returns_none_when_too_few_rows(self):
        df = _make_df(n=100)
        result = compute_signals("TEST", df)
        assert result is None

    def test_returns_signals_with_enough_data(self):
        df = _make_df(n=250)
        result = compute_signals("TEST", df)
        assert result is not None
        assert isinstance(result, TechnicalSignals)

    def test_ticker_preserved(self):
        df = _make_df(n=250)
        result = compute_signals("AAPL", df)
        assert result is not None
        assert result.ticker == "AAPL"

    def test_current_price_matches_last_close(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.current_price == pytest.approx(float(df["Close"].iloc[-1]))

    def test_rsi_in_valid_range(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert 0 <= result.rsi_14 <= 100

    def test_atr_positive(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.atr_14 > 0

    def test_price_above_50sma_consistent(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.price_above_50sma == (result.current_price > result.sma_50)

    def test_price_above_200sma_consistent(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.price_above_200sma == (result.current_price > result.sma_200)

    def test_ema21_above_50sma_consistent(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.ema_21_above_50sma == (result.ema_21 > result.sma_50)

    def test_volume_spike_consistent(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.volume_spike == (result.volume_ratio >= 1.5)

    def test_bb_ordering(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        assert result.bb_lower < result.bb_middle < result.bb_upper

    def test_fib_ordering(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        # fib_61_8 is deeper retracement (lower price) than fib_38_2
        assert result.fib_61_8 < result.fib_38_2

    def test_returns_none_for_none_df(self):
        result = compute_signals("X", None)
        assert result is None

    def test_to_prompt_str_contains_key_fields(self):
        df = _make_df(n=250)
        result = compute_signals("X", df)
        assert result is not None
        text = result.to_prompt_str()
        assert "RSI" in text
        assert "ATR" in text
        assert "Fib" in text
