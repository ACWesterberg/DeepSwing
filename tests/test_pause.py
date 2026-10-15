from __future__ import annotations

from unittest.mock import patch

import pytest

import src.scheduler.scan_loop as scan_loop
from src.portfolio.simulator import get_portfolio, reset_portfolios


@pytest.fixture(autouse=True)
def _clean():
    reset_portfolios()
    yield
    reset_portfolios()


def _open(track: str, ticker: str, market: str, price: float) -> None:
    get_portfolio(track).open_trade(
        ticker=ticker, market=market, quantity=10.0, entry_price=price,
        stop_loss=price * 0.95, target=price * 1.15, regime="trending",
        reasoning="test", confidence=0.8,
    )


def test_liquidate_all_closes_every_track_and_skips_learning():
    _open("claude", "VOLV-B.STO", "nordic", 250.0)
    _open("gpt", "AAPL", "us", 1800.0)
    with patch.object(scan_loop, "_get_current_prices", side_effect=lambda t, m: {"VOLV-B.STO": 240.0} if m == "nordic" else {}), \
         patch.object(scan_loop, "_trigger_erl") as erl, \
         patch.object(scan_loop, "_record_heuristic_outcome") as outcome:
        sold = scan_loop.liquidate_all()

    assert {r["ticker"] for r in sold} == {"VOLV-B.STO", "AAPL"}
    assert {r["ticker"]: r["live_quote"] for r in sold} == {"VOLV-B.STO": True, "AAPL": False}
    for track in ("claude", "gpt"):
        portfolio = get_portfolio(track)
        assert portfolio.open_positions == []
        assert [t.exit_reason for t in portfolio.closed_trades] == ["paused"]
    # AAPL had no quote: sold at its last observed price rather than left open
    assert get_portfolio("gpt").closed_trades[0].exit_price == pytest.approx(1800.0, rel=0.01)
    erl.assert_not_called()
    outcome.assert_not_called()


def test_liquidate_all_with_nothing_open_is_a_noop():
    with patch.object(scan_loop, "_get_current_prices") as prices:
        assert scan_loop.liquidate_all() == []
    prices.assert_not_called()
