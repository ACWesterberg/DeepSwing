from __future__ import annotations

import pytest

from config.settings import settings
from src.portfolio.simulator import get_portfolio, reset_portfolios


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    original = dict(settings.market_allocation)
    yield
    settings.market_allocation = original
    reset_portfolios()


def _open(portfolio, market: str, quantity: float, entry_price: float, ticker: str):
    return portfolio.open_trade(
        ticker=ticker, market=market, quantity=quantity, entry_price=entry_price,
        stop_loss=entry_price * 0.95, target=entry_price * 1.2, regime="trending",
        reasoning="test", confidence=0.8,
    )


class TestMarketExposure:
    def test_exposure_sums_only_matching_market(self):
        p = get_portfolio("claude")
        _open(p, "us", 10.0, 100.0, "AAPL")
        _open(p, "nordic", 5.0, 200.0, "ERIC-B.ST")
        # ~= nominal (fills carry a small slippage markup)
        assert p.market_exposure("us") == pytest.approx(10.0 * 100.0, rel=1e-2)
        assert p.market_exposure("nordic") == pytest.approx(5.0 * 200.0, rel=1e-2)

    def test_exposure_zero_for_empty_market(self):
        p = get_portfolio("claude")
        assert p.market_exposure("us") == 0.0


class TestMarketBudget:
    def test_budget_capped_by_allocation(self):
        settings.market_allocation = {"nordic": 0.5, "us": 0.5}
        p = get_portfolio("claude")  # 100k equity, all cash
        # No positions yet → budget is min(cash, 0.5 * equity) = 50k
        assert p.market_budget_remaining("us") == pytest.approx(50_000.0)

    def test_budget_shrinks_as_market_fills(self):
        settings.market_allocation = {"nordic": 0.5, "us": 0.5}
        p = get_portfolio("claude")
        _open(p, "us", 300.0, 100.0, "AAPL")  # 30k of US exposure
        # allowed 50k of ~100k equity, ~30k used → ~20k remaining (net of costs)
        assert p.market_budget_remaining("us") == pytest.approx(20_000.0, rel=1e-2)

    def test_uncapped_market_limited_by_cash_only(self):
        settings.market_allocation = {"nordic": 0.5}  # us omitted → uncapped
        p = get_portfolio("claude")
        assert p.market_budget_remaining("us") == pytest.approx(p.cash)

    def test_allocation_ge_one_is_uncapped(self):
        settings.market_allocation = {"us": 1.0}
        p = get_portfolio("claude")
        assert p.market_budget_remaining("us") == pytest.approx(p.cash)

    def test_budget_never_negative(self):
        settings.market_allocation = {"us": 0.1}
        p = get_portfolio("claude")
        _open(p, "us", 300.0, 100.0, "AAPL")  # 30k >> 10k cap
        assert p.market_budget_remaining("us") == 0.0


class TestCanOpenInMarket:
    def test_full_us_book_blocks_us_but_allows_nordic(self):
        """The reported bug: a US book at its allocation cap must not starve Nordic.
        The cap keeps ~half the cash reserved, so Nordic still has budget."""
        settings.market_allocation = {"nordic": 0.5, "us": 0.5}
        p = get_portfolio("claude")
        # Fill US up to (just under) its 50% cap; the cap is what keeps cash in
        # reserve for Nordic — without it US would consume everything.
        _open(p, "us", 480.0, 100.0, "AAPL")  # ~48k of a ~50k cap
        assert p.can_open_in_market("us") is False
        # Nordic still has its reserved half available
        assert p.can_open_in_market("nordic") is True

    def test_market_at_cap_blocks_that_market(self):
        settings.market_allocation = {"nordic": 0.5, "us": 0.5}
        p = get_portfolio("claude")
        _open(p, "us", 490.0, 100.0, "AAPL")  # 49k of a 50k cap → under threshold
        assert p.can_open_in_market("us") is False

    def test_open_when_budget_available(self):
        settings.market_allocation = {"nordic": 0.5, "us": 0.5}
        p = get_portfolio("claude")
        assert p.can_open_in_market("us") is True
        assert p.can_open_in_market("nordic") is True
