from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import src.agent.memory as _memory_module
from config.settings import settings
from src.portfolio.simulator import get_portfolio, reset_portfolios


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    reset_portfolios()
    _memory_module._stores.clear()
    monkeypatch.setattr(type(settings), "heuristics_dir", property(lambda self: tmp_path))
    yield
    reset_portfolios()
    _memory_module._stores.clear()


def _do_reset(tmp_path: Path, tracks=("claude", "gpt")):
    """Replicate the reset logic from the /api/reset endpoint."""
    cleared = {}
    for track in tracks:
        heuristic_dir = tmp_path / track
        count = len(list(heuristic_dir.glob("*.json"))) if heuristic_dir.exists() else 0
        if heuristic_dir.exists():
            shutil.rmtree(heuristic_dir)
            heuristic_dir.mkdir(parents=True, exist_ok=True)
        _memory_module._stores.pop(track, None)
        cleared[track] = count
    reset_portfolios()
    return cleared


class TestResetBehavior:
    def test_portfolios_start_at_initial_equity(self):
        """After reset, both portfolios have full starting capital."""
        for track in ("claude", "gpt"):
            p = get_portfolio(track)
            assert p.equity == pytest.approx(settings.starting_capital_sek, rel=1e-6)
            assert p.cash == pytest.approx(settings.starting_capital_sek, rel=1e-6)
            assert p.open_positions == []
            assert p.closed_trades == []

    def test_reset_after_trade_clears_positions(self):
        portfolio = get_portfolio("claude")
        portfolio.open_trade(
            ticker="AAPL", market="us", quantity=5.0, entry_price=100.0,
            stop_loss=95.0, target=115.0, regime="trending",
            reasoning="test", confidence=0.8,
        )
        assert portfolio.cash < settings.starting_capital_sek

        _do_reset(settings.heuristics_dir)

        fresh = get_portfolio("claude")
        assert fresh.cash == pytest.approx(settings.starting_capital_sek, rel=1e-6)
        assert fresh.open_positions == []

    def test_reset_after_closed_trade_clears_history(self):
        portfolio = get_portfolio("gpt")
        portfolio.open_trade(
            ticker="MSFT", market="us", quantity=5.0, entry_price=100.0,
            stop_loss=95.0, target=115.0, regime="trending",
            reasoning="test", confidence=0.7,
        )
        portfolio.update_prices({"MSFT": 90.0})
        assert len(portfolio.closed_trades) == 1

        _do_reset(settings.heuristics_dir)

        fresh = get_portfolio("gpt")
        assert fresh.closed_trades == []

    def test_reset_deletes_heuristic_files(self, tmp_path):
        from src.agent.memory import get_store
        for track in ("claude", "gpt"):
            store = get_store(track)
            store.save(trigger="IF X", action="DO Y", quality_score=6.0)
            store.save(trigger="IF A", action="DO B", quality_score=5.0)

        assert len(list((tmp_path / "claude").glob("*.json"))) == 2
        assert len(list((tmp_path / "gpt").glob("*.json"))) == 2

        cleared = _do_reset(tmp_path)

        assert cleared["claude"] == 2
        assert cleared["gpt"] == 2
        assert list((tmp_path / "claude").glob("*.json")) == []
        assert list((tmp_path / "gpt").glob("*.json")) == []

    def test_reset_clears_heuristic_store_cache(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        store.save(trigger="IF X", action="DO Y", quality_score=6.0)
        assert "claude" in _memory_module._stores

        _do_reset(tmp_path)

        assert "claude" not in _memory_module._stores

    def test_selective_reset_only_clears_target_track(self, tmp_path):
        from src.agent.memory import get_store
        for track in ("claude", "gpt"):
            get_store(track).save(trigger="IF X", action="DO Y", quality_score=5.0)

        _do_reset(tmp_path, tracks=("claude",))

        # Claude cleared, GPT untouched
        assert list((tmp_path / "claude").glob("*.json")) == []
        assert len(list((tmp_path / "gpt").glob("*.json"))) == 1

    def test_reset_reports_correct_heuristic_count(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        for _ in range(3):
            store.save(trigger="IF X", action="DO Y", quality_score=5.0)

        cleared = _do_reset(tmp_path, tracks=("claude",))
        assert cleared["claude"] == 3

    def test_fresh_portfolio_has_correct_peak_equity(self):
        _do_reset(settings.heuristics_dir)
        p = get_portfolio("claude")
        assert p.peak_equity == pytest.approx(settings.starting_capital_sek, rel=1e-6)

    def test_fresh_portfolio_not_in_drawdown_mode(self):
        _do_reset(settings.heuristics_dir)
        p = get_portfolio("claude")
        assert p.is_drawdown_mode is False
