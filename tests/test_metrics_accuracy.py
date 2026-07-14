from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import ClosedTrade, Portfolio


def _trade(trade_id: int, entry: float, exit_: float, stop: float, days_ago: int) -> ClosedTrade:
    now = datetime.utcnow()
    return ClosedTrade(
        trade_id=trade_id, ticker=f"T{trade_id}", market="us", quantity=10.0,
        entry_price=entry, exit_price=exit_, stop_loss=stop, target=entry * 1.2,
        entry_time=now - timedelta(days=days_ago + 2),
        exit_time=now - timedelta(days=days_ago),
        regime="trending", reasoning="", confidence=0.5, exit_reason="take_profit",
    )


class TestAvgRrrIncludesLosers:
    def test_losers_pull_avg_rrr_down(self):
        portfolio = Portfolio("claude")
        # Winner: risk 5, reward 10 → R = +2.0; Loser: risk 5, reward -5 → R = -1.0
        portfolio.closed_trades = [
            _trade(1, 100.0, 110.0, 95.0, days_ago=3),
            _trade(2, 100.0, 95.0, 95.0, days_ago=1),
        ]
        metrics = compute_metrics(portfolio)
        assert metrics.avg_rrr == pytest.approx(0.5)  # (2.0 + -1.0) / 2
        assert metrics.optimization_metric == pytest.approx(0.5)

    def test_equity_curve_reconciles_with_net_pnl(self):
        portfolio = Portfolio("claude")
        position = portfolio.open_trade(
            ticker="AAPL", market="us", quantity=10.0, entry_price=100.0,
            stop_loss=95.0, target=120.0, regime="trending",
            reasoning="test", confidence=0.8,
        )
        portfolio.close_trade(position.trade_id, exit_price=110.0, exit_reason="take_profit")

        from src.portfolio.metrics import _build_equity_curve
        curve = _build_equity_curve(portfolio)
        # All positions closed → realized curve endpoint equals actual cash equity
        assert curve[-1] == pytest.approx(portfolio.equity)
