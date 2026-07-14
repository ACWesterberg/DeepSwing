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


class TestNetPnl:
    def test_pnl_is_net_of_round_trip_commission(self):
        portfolio = Portfolio("claude")
        position = portfolio.open_trade(
            ticker="AAPL", market="us", quantity=10.0, entry_price=100.0,
            stop_loss=95.0, target=120.0, regime="trending",
            reasoning="test", confidence=0.8,
        )
        cash_before_close = portfolio.cash
        closed = portfolio.close_trade(position.trade_id, exit_price=110.0, exit_reason="take_profit")

        gross = (closed.exit_price - closed.entry_price) * closed.quantity
        assert closed.commission > 0
        assert closed.pnl == pytest.approx(gross - closed.commission)
        # Net pnl reconciles with cash across the full round trip
        cash_delta = portfolio.cash - (settings.starting_capital_sek)
        assert cash_delta == pytest.approx(closed.pnl)
        assert cash_before_close < portfolio.cash  # sanity: proceeds landed

    def test_pnl_pct_uses_cost_basis(self):
        portfolio = Portfolio("claude")
        position = portfolio.open_trade(
            ticker="AAPL", market="us", quantity=10.0, entry_price=100.0,
            stop_loss=95.0, target=120.0, regime="trending",
            reasoning="test", confidence=0.8,
        )
        closed = portfolio.close_trade(position.trade_id, exit_price=110.0, exit_reason="take_profit")
        assert closed.pnl_pct == pytest.approx(closed.pnl / (closed.entry_price * closed.quantity))

    def test_pre_upgrade_trades_default_to_gross(self):
        from src.portfolio.simulator import ClosedTrade
        from datetime import datetime
        # Persisted state from before the commission field existed
        legacy = ClosedTrade.from_state({
            "trade_id": 1, "ticker": "AAPL", "market": "us", "quantity": 10.0,
            "entry_price": 100.0, "exit_price": 110.0, "stop_loss": 95.0,
            "target": 120.0, "entry_time": datetime.utcnow().isoformat(),
            "exit_time": datetime.utcnow().isoformat(),
            "regime": "trending", "reasoning": "", "confidence": 0.5,
            "exit_reason": "take_profit",
        })
        assert legacy.commission == 0.0
        assert legacy.pnl == pytest.approx(100.0)
