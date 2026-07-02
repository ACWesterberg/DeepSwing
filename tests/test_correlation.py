from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent.risk import compute_return_correlations, validate_trade
from tests.test_risk import EQUITY, VALID_ENTRY, VALID_STOP, VALID_TARGET, _make_signals


def _df_from_prices(prices: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=idx)


def _random_walk(seed: int, n: int = 120) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))


class TestComputeReturnCorrelations:
    def test_identical_series_fully_correlated(self):
        prices = _random_walk(seed=1)
        df = _df_from_prices(prices)
        corr = compute_return_correlations(df, ["OTHER"], {"OTHER": df.copy()})
        assert corr["OTHER"] == pytest.approx(1.0)

    def test_inverse_series_negatively_correlated(self):
        prices = _random_walk(seed=1)
        df = _df_from_prices(prices)
        inverse = _df_from_prices(prices[0] * (prices[0] / prices))
        corr = compute_return_correlations(df, ["OTHER"], {"OTHER": inverse})
        assert corr["OTHER"] < -0.9

    def test_independent_series_low_correlation(self):
        df_a = _df_from_prices(_random_walk(seed=1))
        df_b = _df_from_prices(_random_walk(seed=2))
        corr = compute_return_correlations(df_a, ["OTHER"], {"OTHER": df_b})
        assert abs(corr["OTHER"]) < 0.5

    def test_missing_ticker_skipped(self):
        df = _df_from_prices(_random_walk(seed=1))
        assert compute_return_correlations(df, ["GONE"], {}) == {}

    def test_no_open_positions_returns_empty(self):
        df = _df_from_prices(_random_walk(seed=1))
        assert compute_return_correlations(df, [], {"X": df}) == {}

    def test_none_candidate_df_returns_empty(self):
        assert compute_return_correlations(None, ["X"], {}) == {}

    def test_insufficient_overlap_skipped(self):
        df_a = _df_from_prices(_random_walk(seed=1))
        short = _df_from_prices(_random_walk(seed=2, n=10))
        assert compute_return_correlations(df_a, ["OTHER"], {"OTHER": short}) == {}

    def test_garbage_data_never_raises(self):
        from unittest.mock import MagicMock
        assert compute_return_correlations(MagicMock(), ["X"], {"X": MagicMock()}) == {}


class TestCorrelationCapInRisk:
    def _validate(self, correlations):
        return validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[{"ticker": "MSFT", "sector": "Technology"}],
            signals=_make_signals(ticker="NVDA"),
            position_correlations=correlations,
        )

    def test_high_correlation_rejected(self):
        result = self._validate({"MSFT": 0.85})
        assert result.approved is False
        assert "correlation" in result.rejection_reason.lower()
        assert "MSFT" in result.rejection_reason

    def test_correlation_at_limit_allowed(self):
        assert self._validate({"MSFT": 0.70}).approved is True

    def test_low_correlation_allowed(self):
        assert self._validate({"MSFT": 0.30}).approved is True

    def test_negative_correlation_allowed(self):
        assert self._validate({"MSFT": -0.90}).approved is True

    def test_worst_of_many_positions_drives_rejection(self):
        result = self._validate({"MSFT": 0.20, "GOOGL": 0.95})
        assert result.approved is False
        assert "GOOGL" in result.rejection_reason

    def test_none_correlations_skips_check(self):
        assert self._validate(None).approved is True
