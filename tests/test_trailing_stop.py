from __future__ import annotations

import pytest

from src.portfolio.simulator import get_portfolio, reset_portfolios


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    yield
    reset_portfolios()


def _open(portfolio, trail_distance: float = 0.0):
    return portfolio.open_trade(
        ticker="AAPL", market="us", quantity=10.0, entry_price=100.0,
        stop_loss=95.0, target=120.0, regime="trending",
        reasoning="test", confidence=0.8, trail_distance=trail_distance,
    )


class TestTrailingStop:
    def test_trail_uses_atr_distance(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=6.0)

        portfolio.update_prices({"AAPL": 110.0})
        pos = portfolio.open_positions[0]
        assert pos.trailing_stop == pytest.approx(104.0)  # 110 - 6

    def test_trail_never_moves_down(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=6.0)

        portfolio.update_prices({"AAPL": 110.0})
        portfolio.update_prices({"AAPL": 106.0})  # pullback above trail
        pos = portfolio.open_positions[0]
        assert pos.trailing_stop == pytest.approx(104.0)

    def test_trailed_exit_labeled_trailing_stop(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=6.0)

        portfolio.update_prices({"AAPL": 112.0})   # trail → 106
        closed = portfolio.update_prices({"AAPL": 105.0})
        assert len(closed) == 1
        assert closed[0].exit_reason == "trailing_stop"
        assert closed[0].pnl > 0

    def test_original_stop_exit_labeled_stop_loss(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=6.0)

        closed = portfolio.update_prices({"AAPL": 94.0})  # straight through stop
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"

    def test_legacy_position_falls_back_to_2pct_trail(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=0.0)  # persisted pre-trail_distance

        portfolio.update_prices({"AAPL": 110.0})
        pos = portfolio.open_positions[0]
        assert pos.trailing_stop == pytest.approx(110.0 * 0.98)

    def test_trail_distance_survives_state_roundtrip(self):
        portfolio = get_portfolio("claude")
        _open(portfolio, trail_distance=6.0)

        state = portfolio.export_state()
        fresh = get_portfolio("gpt")
        fresh.import_state(state)
        assert fresh.open_positions[0].trail_distance == pytest.approx(6.0)


class TestPeakEquityRatchet:
    def test_peak_updates_on_mark_to_market(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        start_peak = portfolio.peak_equity

        portfolio.update_prices({"AAPL": 115.0})  # below target, stays open
        assert portfolio.peak_equity > start_peak
