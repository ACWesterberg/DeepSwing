from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import (
    BacktestEngine,
    BacktestTrade,
    _compute_metrics,
    _empty_metrics,
    _get_prices_for_day,
    _SimPortfolio,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _trending_df(n: int = 300, start_price: float = 100.0) -> pd.DataFrame:
    """Strongly uptrending OHLCV DataFrame (passes the screener)."""
    rng = np.random.default_rng(0)
    prices = start_price + np.linspace(0, 30, n) + rng.normal(0, 0.3, n)
    prices = np.clip(prices, 1, None)
    high = prices * (1 + rng.uniform(0.001, 0.01, n))
    low = prices * (1 - rng.uniform(0.001, 0.01, n))
    volume = rng.integers(500_000, 2_000_000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"Open": prices, "High": high, "Low": low, "Close": prices, "Volume": volume}, index=idx)


def _make_closed_trade(pnl_pct: float, entry: float = 100.0) -> BacktestTrade:
    exit_price = entry * (1 + pnl_pct)
    stop = entry * 0.97
    return BacktestTrade(
        ticker="TEST",
        entry_date=date(2024, 1, 1),
        entry_price=entry,
        exit_date=date(2024, 1, 10),
        exit_price=exit_price,
        exit_reason="take_profit" if pnl_pct > 0 else "stop_loss",
        stop_loss=stop,
        target=entry * 1.09,
        quantity=10.0,
    )


# ---------------------------------------------------------------------------
# BacktestTrade
# ---------------------------------------------------------------------------

class TestBacktestTrade:
    def test_pnl_positive_win(self):
        t = _make_closed_trade(0.05)
        assert t.pnl > 0

    def test_pnl_negative_loss(self):
        t = _make_closed_trade(-0.03)
        assert t.pnl < 0

    def test_pnl_pct_correct(self):
        t = _make_closed_trade(0.10)
        assert t.pnl_pct == pytest.approx(0.10, rel=1e-6)

    def test_rrr_achieved_positive_trade(self):
        t = _make_closed_trade(0.09)  # entry=100, stop=97, exit=109 → rrr = 9/3 = 3
        assert t.rrr_achieved == pytest.approx(3.0, rel=1e-3)

    def test_to_dict_has_required_keys(self):
        t = _make_closed_trade(0.05)
        d = t.to_dict()
        for key in ("ticker", "entry_date", "entry_price", "exit_price", "pnl", "pnl_pct", "rrr_achieved"):
            assert key in d

    def test_open_trade_pnl_is_zero(self):
        t = BacktestTrade(
            ticker="X", entry_date=date(2024, 1, 1), entry_price=100.0,
            exit_date=None, exit_price=None, exit_reason="open",
            stop_loss=97.0, target=109.0, quantity=5.0,
        )
        assert t.pnl == 0.0


# ---------------------------------------------------------------------------
# _SimPortfolio
# ---------------------------------------------------------------------------

class TestSimPortfolio:
    def test_initial_equity_equals_cash(self):
        p = _SimPortfolio(100_000)
        assert p.equity == 100_000

    def test_open_position_reduces_cash(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        assert p.cash == pytest.approx(100_000 - 10_000, rel=1e-6)

    def test_has_ticker(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        assert p.has_ticker("AAPL")
        assert not p.has_ticker("MSFT")

    def test_stop_hit_closes_position(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": 96.0}, date(2024, 1, 5))
        assert not p.has_ticker("AAPL")
        assert len(p.closed_trades) == 1
        assert p.closed_trades[0].exit_reason == "stop_loss"

    def test_target_hit_closes_position(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": 110.0}, date(2024, 1, 5))
        assert not p.has_ticker("AAPL")
        assert p.closed_trades[0].exit_reason == "take_profit"

    def test_insufficient_cash_blocks_open(self):
        p = _SimPortfolio(500)  # very little cash
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        assert not p.has_ticker("AAPL")

    def test_peak_equity_updated_after_profitable_close(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": 110.0}, date(2024, 1, 5))
        assert p.peak_equity >= 100_000

    def test_drawdown_mode_false_at_start(self):
        p = _SimPortfolio(100_000)
        assert p.is_drawdown_mode is False


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_empty_trade_list_returns_empty_metrics(self):
        m = _compute_metrics([], 100_000)
        assert m == _empty_metrics()

    def test_win_rate_correct(self):
        trades = [_make_closed_trade(0.05), _make_closed_trade(0.05), _make_closed_trade(-0.03)]
        m = _compute_metrics(trades, 100_000)
        assert m["win_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_total_pnl_correct(self):
        trades = [_make_closed_trade(0.05), _make_closed_trade(-0.03)]
        m = _compute_metrics(trades, 100_000)
        expected = sum(t.pnl for t in trades)
        assert m["total_pnl"] == pytest.approx(expected, rel=1e-3)

    def test_optimization_metric_is_win_rate_times_avg_rrr(self):
        trades = [_make_closed_trade(0.09), _make_closed_trade(-0.03)]
        m = _compute_metrics(trades, 100_000)
        assert m["optimization_metric"] == pytest.approx(m["win_rate"] * m["avg_rrr"], rel=1e-3)

    def test_max_drawdown_is_non_negative(self):
        trades = [_make_closed_trade(-0.05), _make_closed_trade(-0.05)]
        m = _compute_metrics(trades, 100_000)
        assert m["max_drawdown_pct"] >= 0

    def test_end_of_window_trades_excluded_from_win_rate(self):
        """Positions forced-closed at window end shouldn't count as losses."""
        normal = _make_closed_trade(0.05)
        eow = BacktestTrade(
            ticker="X", entry_date=date(2024, 1, 1), entry_price=100.0,
            exit_date=date(2024, 3, 31), exit_price=105.0, exit_reason="end_of_window",
            stop_loss=97.0, target=109.0, quantity=10.0,
        )
        m = _compute_metrics([normal, eow], 100_000)
        assert m["total_trades"] == 1  # only the normal close counted


# ---------------------------------------------------------------------------
# BacktestEngine (with mocked data loading)
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def _make_engine(self, n_windows: int = 1) -> BacktestEngine:
        return BacktestEngine(
            market="us",
            tickers=["AAPL"],
            start=date(2024, 6, 1),
            end=date(2024, 12, 31),
            initial_equity=100_000.0,
            n_windows=n_windows,
        )

    def _mock_ohlcv(self, df: pd.DataFrame) -> dict:
        return {"AAPL": df}

    def test_returns_backtest_result(self):
        engine = self._make_engine()
        df = _trending_df(n=350)
        with patch.object(engine, "_load_data", return_value=self._mock_ohlcv(df)):
            result = engine.run()
        assert result.market == "us"
        assert result.n_windows == 1
        assert "overall_metrics" in result.to_dict()

    def test_empty_data_returns_empty_result(self):
        engine = self._make_engine()
        with patch.object(engine, "_load_data", return_value={}):
            result = engine.run()
        assert result.windows == []

    def test_split_windows_count(self):
        engine = self._make_engine(n_windows=4)
        windows = engine._split_windows()
        assert len(windows) == 4

    def test_split_windows_cover_full_range(self):
        engine = self._make_engine(n_windows=3)
        windows = engine._split_windows()
        assert windows[0][0] == engine.start
        assert windows[-1][1] == engine.end

    def test_multiple_windows_run(self):
        engine = self._make_engine(n_windows=2)
        df = _trending_df(n=350)
        with patch.object(engine, "_load_data", return_value=self._mock_ohlcv(df)):
            result = engine.run()
        assert len(result.windows) == 2

    def test_overall_metrics_present(self):
        engine = self._make_engine()
        df = _trending_df(n=350)
        with patch.object(engine, "_load_data", return_value=self._mock_ohlcv(df)):
            result = engine.run()
        keys = {"total_trades", "win_rate", "avg_rrr", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"}
        assert keys.issubset(result.overall_metrics.keys())

    def test_to_dict_serializable(self):
        engine = self._make_engine()
        df = _trending_df(n=350)
        with patch.object(engine, "_load_data", return_value=self._mock_ohlcv(df)):
            result = engine.run()
        d = result.to_dict()
        import json
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# _get_prices_for_day
# ---------------------------------------------------------------------------

class TestGetPricesForDay:
    def test_returns_price_for_matching_day(self):
        df = _trending_df(n=10)
        target_day = df.index[5].date()
        prices = _get_prices_for_day({"AAPL": df}, ["AAPL"], target_day)
        assert "AAPL" in prices
        assert prices["AAPL"] == pytest.approx(float(df.iloc[5]["Close"]), rel=1e-6)

    def test_returns_empty_for_missing_ticker(self):
        df = _trending_df(n=10)
        target_day = df.index[0].date()
        prices = _get_prices_for_day({"AAPL": df}, ["MSFT"], target_day)
        assert prices == {}

    def test_returns_empty_for_missing_day(self):
        df = _trending_df(n=10)
        prices = _get_prices_for_day({"AAPL": df}, ["AAPL"], date(2000, 1, 1))
        assert prices == {}
