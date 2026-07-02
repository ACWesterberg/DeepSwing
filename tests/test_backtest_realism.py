from __future__ import annotations

from datetime import date

import pytest

from src.backtesting.engine import _SimPortfolio


def _bar(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


class TestCosts:
    def test_slippage_and_commission_applied_at_open(self):
        p = _SimPortfolio(100_000, commission_rate=0.001, slippage=0.0005)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))

        fill = 100.0 * 1.0005
        cost = fill * 100.0
        commission = cost * 0.001
        assert p.cash == pytest.approx(100_000 - cost - commission)
        assert p.total_commission == pytest.approx(commission)
        assert p._positions["AAPL"].entry_price == pytest.approx(fill)

    def test_exit_costs_reduce_proceeds(self):
        p = _SimPortfolio(100_000, commission_rate=0.001, slippage=0.0005)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": 110.0}, date(2024, 1, 5))

        closed = p.closed_trades[0]
        assert closed.exit_price == pytest.approx(110.0 * (1 - 0.0005))
        assert closed.commission > 0
        assert closed.net_pnl < closed.pnl

    def test_zero_cost_defaults_keep_old_arithmetic(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        assert p.cash == pytest.approx(90_000)


class TestIntradayExits:
    def test_low_through_stop_fires_even_when_close_recovers(self):
        # Close-only checking would miss this: closes at 100, but low traded 96
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(100, 101, 96, 100)}, date(2024, 1, 2))

        assert not p.has_ticker("AAPL")
        closed = p.closed_trades[0]
        assert closed.exit_reason == "stop_loss"
        assert closed.exit_price == pytest.approx(97.0)  # filled at the stop

    def test_gap_down_fills_at_open_not_stop(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(94, 95, 93, 94)}, date(2024, 1, 2))

        assert p.closed_trades[0].exit_price == pytest.approx(94.0)

    def test_high_through_target_fires(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(105, 110, 104, 106)}, date(2024, 1, 2))

        closed = p.closed_trades[0]
        assert closed.exit_reason == "take_profit"
        assert closed.exit_price == pytest.approx(109.0)

    def test_both_hit_same_bar_stop_wins(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 109.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(103, 110, 96, 108)}, date(2024, 1, 2))

        assert p.closed_trades[0].exit_reason == "stop_loss"


class TestBacktestTrailingStop:
    def test_trail_raises_stop_and_labels_exit(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 120.0, 100.0, date(2024, 1, 1), trail_distance=6.0)

        p.update({"AAPL": _bar(109, 111, 108, 110)}, date(2024, 1, 2))  # trail → 104
        assert p._positions["AAPL"].trailing_stop == pytest.approx(104.0)

        p.update({"AAPL": _bar(105, 106, 103, 104)}, date(2024, 1, 3))
        closed = p.closed_trades[0]
        assert closed.exit_reason == "trailing_stop"
        assert closed.exit_price == pytest.approx(104.0)
        assert closed.pnl > 0

    def test_trail_not_raised_before_exit_checks_same_bar(self):
        # The bar that would raise the trail (close 110) also dips to 103;
        # yesterday's stop is 97, so no exit — the trail must not fire same-bar.
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 120.0, 100.0, date(2024, 1, 1), trail_distance=6.0)
        p.update({"AAPL": _bar(104, 111, 103, 110)}, date(2024, 1, 2))

        assert p.has_ticker("AAPL")
        assert p._positions["AAPL"].trailing_stop == pytest.approx(104.0)

    def test_no_trailing_without_trail_distance(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 97.0, 120.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(109, 111, 108, 110)}, date(2024, 1, 2))
        assert p._positions["AAPL"].trailing_stop is None


class TestMarkToMarket:
    def test_equity_reflects_current_price(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 90.0, 200.0, 100.0, date(2024, 1, 1))
        p.update({"AAPL": _bar(110, 112, 108, 110)}, date(2024, 1, 2))

        assert p.equity == pytest.approx(90_000 + 110.0 * 100.0)
        assert p.peak_equity == pytest.approx(p.equity)

    def test_drawdown_mode_sees_open_losses(self):
        p = _SimPortfolio(100_000)
        p.open_position("AAPL", 100.0, 50.0, 300.0, 500.0, date(2024, 1, 1))  # 50k position
        p.update({"AAPL": _bar(75, 76, 74, 75)}, date(2024, 1, 2))  # -12.5k open loss

        assert p.is_drawdown_mode is True
