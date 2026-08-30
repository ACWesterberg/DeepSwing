from __future__ import annotations

import pytest

from config.settings import settings
from src.agent.risk import validate_trade
from src.portfolio.simulator import get_portfolio, reset_portfolios
from tests.test_risk import _make_signals


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    yield
    reset_portfolios()


def _position_frac(atr_pct: float) -> float:
    """Position value as a fraction of equity, after the value cap."""
    stop_frac = settings.atr_stop_multiplier * atr_pct
    return min(settings.max_risk_per_trade / stop_frac, settings.max_position_pct)


def _slots(market: str, atr_pct: float = 0.03) -> int:
    """How many positions fit inside a market's allocation budget."""
    cap = settings.market_allocation.get(market)
    if cap is None or cap >= 1.0:
        return 999
    return int(cap / _position_frac(atr_pct))


class TestBookCapacity:
    """Concurrent positions is the throughput ceiling on learning: closed trades
    are the scarce input to both MIPRO and ERL, and slots x turnover is how fast
    they accumulate. At max_position_pct 0.25 the whole book held two."""

    def test_every_configured_market_can_hold_a_position(self):
        # A 25% position could not fit the 20% EU budget, so EU was structurally
        # untradeable — zero entries possible, ever, and nothing logged it.
        for market in settings.market_allocation:
            assert _slots(market) >= 1, (
                f"{market} (cap {settings.market_allocation[market]:.0%}) cannot fit a "
                f"{_position_frac(0.03):.1%} position — that market can never trade"
            )

    def test_book_holds_enough_positions_to_learn(self):
        total = sum(_slots(m) for m in settings.market_allocation)
        # 30 closed trades are needed before MIPRO runs at all. At a ~9-day hold,
        # fewer than ~6 slots pushes that past two months.
        assert total >= 6, f"only {total} concurrent positions fit across all markets"

    @pytest.mark.parametrize("atr_pct", [0.02, 0.03, 0.04, 0.06])
    def test_position_never_exceeds_the_cap(self, atr_pct):
        assert _position_frac(atr_pct) <= settings.max_position_pct + 1e-9

    def test_sizing_respects_the_cap_end_to_end(self):
        # The arithmetic above mirrors risk.py; this pins them together so the
        # capacity guarantees above stay honest if sizing changes.
        entry = 100.0
        atr_pct = 0.03
        stop = entry * (1 - settings.atr_stop_multiplier * atr_pct)
        result = validate_trade(
            action="BUY",
            entry_price=entry,
            stop_loss=stop,
            target=entry + (settings.min_rrr + 0.5) * (entry - stop),
            portfolio_equity=100_000.0,
            open_positions=[],
            signals=_make_signals(atr_14=entry * atr_pct),
        )
        assert result.approved is True
        actual = result.quantity * entry / 100_000.0
        assert actual == pytest.approx(_position_frac(atr_pct), rel=1e-3)


class TestMarketBudgetHonoursCapacity:
    def test_a_market_accepts_multiple_positions_before_filling(self):
        portfolio = get_portfolio("claude")
        market = "nordic"
        expected = _slots(market)
        assert expected >= 2, "nordic should fit more than a single position"

        size = _position_frac(0.03) * portfolio.equity
        opened = 0
        for i in range(expected):
            if not portfolio.can_open_in_market(market):
                break
            qty = size / 100.0
            portfolio.open_trade(
                ticker=f"TICK{i}.ST", market=market, quantity=qty, entry_price=100.0,
                stop_loss=95.5, target=112.0, regime="trending",
                reasoning="t", confidence=0.7, trail_distance=6.0,
            )
            opened += 1

        assert opened >= 2, (
            f"market budget allowed only {opened} position(s) where {expected} should fit"
        )
