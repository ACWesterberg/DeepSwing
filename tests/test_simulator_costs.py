from __future__ import annotations

import pytest

from config.settings import settings
from src.portfolio.simulator import Portfolio


class TestFxCommissionParity:
    def _open(self, portfolio: Portfolio, market: str):
        return portfolio.open_trade(
            ticker="0RQ6.L" if market == "eu" else "AAPL",
            market=market,
            quantity=10.0,
            entry_price=100.0,
            stop_loss=95.0,
            target=120.0,
            regime="trending",
            reasoning="test",
            confidence=0.8,
        )

    def test_eu_close_charges_fx_commission_like_open(self):
        portfolio = Portfolio("claude")
        position = self._open(portfolio, "eu")
        commission_at_open = portfolio.total_commission

        portfolio.close_trade(position.trade_id, exit_price=110.0, exit_reason="take_profit")

        proceeds = 110.0 * (1 - settings.simulated_slippage) * 10.0
        expected = proceeds * (settings.commission_pct + settings.fx_commission_pct)
        assert portfolio.total_commission - commission_at_open == pytest.approx(expected)

    def test_us_close_charges_fx_commission(self):
        portfolio = Portfolio("claude")
        position = self._open(portfolio, "us")
        commission_at_open = portfolio.total_commission

        portfolio.close_trade(position.trade_id, exit_price=110.0, exit_reason="take_profit")

        proceeds = 110.0 * (1 - settings.simulated_slippage) * 10.0
        expected = proceeds * (settings.commission_pct + settings.fx_commission_pct)
        assert portfolio.total_commission - commission_at_open == pytest.approx(expected)
