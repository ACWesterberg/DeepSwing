from __future__ import annotations

import pytest

from config.settings import settings
from src.portfolio.simulator import (
    OpenPosition,
    breakeven_price,
    get_portfolio,
    reset_portfolios,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    yield
    reset_portfolios()


def _open(portfolio, trail_distance: float = 8.0, market: str = "us"):
    """Entry 100, ATR 4 (trail 2xATR = 8), stop 94 = 1.5xATR below entry."""
    return portfolio.open_trade(
        ticker="AAPL", market=market, quantity=10.0, entry_price=100.0,
        stop_loss=94.0, target=112.0, regime="trending",
        reasoning="test", confidence=0.8, trail_distance=trail_distance,
    )


class TestArming:
    def test_does_not_arm_below_threshold(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        # arms at +1 ATR = 104; 103 is short of it
        portfolio.update_prices({"AAPL": 103.0})
        assert portfolio.open_positions[0].breakeven_armed is False

    def test_arms_at_threshold(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 104.5})
        assert portfolio.open_positions[0].breakeven_armed is True

    def test_stays_armed_after_pullback(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 105.0})
        portfolio.update_prices({"AAPL": 101.0})  # above the floor, stays open
        pos = portfolio.open_positions[0]
        assert pos.breakeven_armed is True

    def test_disabled_by_zero_multiplier(self, monkeypatch):
        monkeypatch.setattr(settings, "breakeven_arm_atr_multiplier", 0.0)
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 110.0})
        assert portfolio.open_positions[0].breakeven_armed is False


class TestBreakevenExit:
    def test_armed_reversal_exits_at_breakeven_not_a_loss(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 105.0})          # arms the floor
        closed = portfolio.update_prices({"AAPL": 99.0})  # reverses through it

        assert len(closed) == 1
        assert closed[0].exit_reason == "breakeven_stop"
        # The fill is the observed mark, not the floor — a 15-min sweep cannot
        # fill at a level it never saw — so this is far better than the -1R
        # stop but not exactly zero. Overshoot below the floor is the same
        # gap-slippage that makes real stop_loss exits average worse than -1R.
        assert closed[0].rrr_achieved > -0.3   # a full stop-out is -1.0

    def test_floor_covers_round_trip_costs(self):
        entry_fill = 100.0 * (1 + settings.simulated_slippage)
        floor = breakeven_price(entry_fill, "us")
        assert floor > entry_fill  # strictly above entry, or costs eat the trade

        portfolio = get_portfolio("claude")
        pos = portfolio.open_trade(
            ticker="AAPL", market="us", quantity=10.0, entry_price=100.0,
            stop_loss=94.0, target=112.0, regime="trending",
            reasoning="test", confidence=0.8, trail_distance=8.0,
        )
        portfolio.update_prices({"AAPL": 105.0})
        closed = portfolio.update_prices({"AAPL": floor})
        assert closed[0].exit_reason == "breakeven_stop"
        assert closed[0].pnl >= 0.0

    def test_unarmed_trade_still_labels_stop_loss(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        closed = portfolio.update_prices({"AAPL": 93.0})  # never profited
        assert closed[0].exit_reason == "stop_loss"

    def test_trail_above_breakeven_still_labels_trailing_stop(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 111.0})          # trail → 103
        closed = portfolio.update_prices({"AAPL": 102.0})
        assert closed[0].exit_reason == "trailing_stop"
        assert closed[0].pnl > 0


class TestRegression:
    """The exact failure observed in the live DB before this fix."""

    def test_ratchet_then_reverse_is_no_longer_a_trailed_winner(self):
        # Entry 100, ATR 4, stop 94. Peak 102.5 (+0.6 ATR) ratcheted the old
        # trail to 94.5 (> stop_loss), so the old predicate labeled this
        # -0.92R exit "trailing_stop". It peaked below the +1 ATR arm, so it
        # is not rescued — but it is now honestly labeled a stop_loss, which
        # is what ERL needs to attribute it to entry quality.
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 102.5})
        closed = portfolio.update_prices({"AAPL": 94.5})

        assert len(closed) == 1
        assert closed[0].exit_reason == "stop_loss"
        assert closed[0].rrr_achieved < -0.5

    def test_ratchet_above_arm_then_reverse_is_rescued(self):
        # Same shape, but the peak clears +1 ATR: the floor arms and converts
        # what was a -1R loss labeled "trailing_stop" into a scratch.
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 104.5})
        closed = portfolio.update_prices({"AAPL": 100.2})

        assert len(closed) == 1
        assert closed[0].exit_reason == "breakeven_stop"
        assert closed[0].rrr_achieved > -0.1

    def test_sub_arming_reversal_is_an_honest_stop_loss(self):
        # Peak +0.6 ATR: under the old code the trail ratcheted above the stop
        # and mislabeled a full -1R loss as a trailed exit.
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 102.4})   # +0.6 ATR, below the arm
        closed = portfolio.update_prices({"AAPL": 94.0})

        assert closed[0].exit_reason == "stop_loss"
        # Slightly worse than -1R: rrr_achieved is net, so a stop-out also
        # carries both commission legs and both slippage legs.
        assert -1.15 < closed[0].rrr_achieved < -1.0


class TestPersistence:
    def test_breakeven_armed_survives_state_roundtrip(self):
        portfolio = get_portfolio("claude")
        _open(portfolio)
        portfolio.update_prices({"AAPL": 105.0})
        assert portfolio.open_positions[0].breakeven_armed is True

        state = portfolio.export_state()
        fresh = get_portfolio("gpt")
        fresh.import_state(state)
        assert fresh.open_positions[0].breakeven_armed is True

    def test_legacy_position_without_key_defaults_to_unarmed(self):
        # Every position already persisted on the Pi lacks this key. A bare
        # d["breakeven_armed"] would KeyError inside restore_portfolios' broad
        # except and silently reset every track to starting capital.
        legacy = {
            "trade_id": 1, "ticker": "AAPL", "market": "us", "quantity": 10.0,
            "entry_price": 100.0, "stop_loss": 94.0, "target": 112.0,
            "entry_time": "2026-08-01T09:00:00",
        }
        pos = OpenPosition.from_state(legacy)
        assert pos.breakeven_armed is False
