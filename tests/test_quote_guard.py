from __future__ import annotations

import pytest

from src.portfolio.simulator import get_portfolio, reset_portfolios


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    yield
    reset_portfolios()


def _open(portfolio):
    """Entry ~100 (slippage), stop 94, target 112."""
    return portfolio.open_trade(
        ticker="NTN.L", market="eu", quantity=10.0, entry_price=100.0,
        stop_loss=94.0, target=112.0, regime="trending",
        reasoning="test", confidence=0.8, trail_distance=8.0,
    )


class TestUnitFlip:
    def test_pence_pound_flip_never_books_an_exit(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        assert portfolio.update_prices({"NTN.L": 1.0}) == []
        assert portfolio.update_prices({"NTN.L": 1.0}) == []
        assert len(portfolio.open_positions) == 1

    def test_flip_upward_is_ignored_too(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        assert portfolio.update_prices({"NTN.L": 10_000.0}) == []
        assert portfolio.open_positions[0].current_price < 200


class TestLargeJump:
    def test_single_bad_print_is_held_back(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        assert portfolio.update_prices({"NTN.L": 40.0}) == []
        # the next quote is back to normal: the print never happened
        assert portfolio.update_prices({"NTN.L": 99.0}) == []
        assert portfolio.open_positions[0].current_price == 99.0

    def test_confirmed_crash_is_booked(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        assert portfolio.update_prices({"NTN.L": 40.0}) == []
        closed = portfolio.update_prices({"NTN.L": 41.0})
        assert len(closed) == 1 and closed[0].exit_reason == "stop_loss"

    def test_ordinary_moves_are_not_delayed(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        closed = portfolio.update_prices({"NTN.L": 93.0})
        assert len(closed) == 1
