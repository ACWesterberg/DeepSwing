from __future__ import annotations

from datetime import datetime, timedelta

from src.portfolio.metrics import build_equity_curve_chart_data
from src.data.universe import get_exchange_for_ticker
from src.portfolio.simulator import ClosedTrade, OpenPosition, Portfolio


class TestEquityCurveData:
    def test_points_are_chronological_iso_timestamps(self):
        portfolio = Portfolio("claude")
        t0 = datetime(2026, 7, 1, 10, 0)
        t1 = t0 + timedelta(days=1)
        portfolio.closed_trades.append(
            ClosedTrade(
                trade_id=1,
                ticker="AAPL",
                market="us",
                quantity=10,
                entry_price=100.0,
                exit_price=95.0,
                stop_loss=90.0,
                target=120.0,
                entry_time=t0,
                exit_time=t1,
                regime="trending",
                reasoning="test",
                confidence=0.8,
                exit_reason="stop_loss",
            )
        )
        portfolio.open_positions.append(
            OpenPosition(
                trade_id=2,
                ticker="VOLV-B.ST",
                market="nordic",
                quantity=5,
                entry_price=200.0,
                stop_loss=180.0,
                target=240.0,
                entry_time=t0 + timedelta(hours=2),
                current_price=190.0,
            )
        )

        points = build_equity_curve_chart_data(portfolio)
        dates = [datetime.fromisoformat(p["date"]) for p in points]
        assert dates == sorted(dates)
        assert all("start" not in p["date"] and p["date"] != "now" for p in points)
        assert points[-1]["equity"] == round(portfolio.equity, 2)


class TestExchangeLookup:
    def test_us_ticker_from_universe(self):
        assert get_exchange_for_ticker("AAPL", "us") == "NASDAQ"

    def test_nordic_suffix_fallback(self):
        assert get_exchange_for_ticker("FOO-B.ST", "nordic") == "OMXS"

    def test_unknown_us_falls_back_to_market(self):
        assert get_exchange_for_ticker("ZZZZZ", "us") == "US"
