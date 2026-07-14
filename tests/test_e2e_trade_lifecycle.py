from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import src.agent.memory as _memory_module
from config.settings import settings
from src.agent.erl import run_erl
from src.portfolio.simulator import get_portfolio, reset_portfolios

# ---------------------------------------------------------------------------
# ERL response fixtures
# ---------------------------------------------------------------------------

_GOOD_ERL_RESPONSE = """\
Trigger: RSI crossed below 50 while price traded below the 20-day EMA
Action: Avoid entering new longs; wait for RSI to recover above 50 before re-entry
Quality: 7
Market: us
Regime: trending
"""

_LOW_QUALITY_ERL_RESPONSE = """\
Trigger: Some vague condition
Action: Do something general
Quality: 1
Market: both
Regime: any
"""

_UNPARSEABLE_ERL_RESPONSE = "No clear lesson can be extracted from this trade."


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state(tmp_path, monkeypatch):
    """Reset portfolio + heuristic store before each test; redirect heuristics to tmp_path."""
    reset_portfolios()
    _memory_module._stores.clear()
    monkeypatch.setattr(type(settings), "heuristics_dir", property(lambda self: tmp_path))
    yield
    reset_portfolios()
    _memory_module._stores.clear()


def _open_position(portfolio, ticker: str = "AAPL", snapshot: str = "RSI=55.2, EMA20=above, ATR=1.5"):
    return portfolio.open_trade(
        ticker=ticker,
        market="us",
        quantity=10.0,
        entry_price=100.0,
        stop_loss=95.0,
        target=115.0,
        regime="trending",
        reasoning="Strong uptrend continuation",
        confidence=0.85,
        technical_snapshot=snapshot,
        sector="Technology",
    )


def _build_erl_trade_dict(closed) -> dict:
    """Mirror what scan_loop._trigger_erl does when building the trade dict for ERL."""
    d = closed.to_dict()
    d["id"] = closed.trade_id
    d["stop_hit"] = closed.exit_reason == "stop_loss"
    d["pnl_pct"] = closed.pnl_pct   # fractional, not percent — to_dict() gives percent
    return d


# ---------------------------------------------------------------------------
# End-to-end: full pipeline
# ---------------------------------------------------------------------------

class TestTradeLifecycleE2E:
    def test_stop_hit_creates_heuristic_file(self, tmp_path):
        portfolio = get_portfolio("claude")
        assert _open_position(portfolio) is not None

        closed_trades = portfolio.update_prices({"AAPL": 90.0})
        assert len(closed_trades) == 1
        closed = closed_trades[0]
        assert closed.exit_reason == "stop_loss"

        with patch("src.agent.erl._call_model", return_value=_GOOD_ERL_RESPONSE):
            heuristic_id = run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )

        assert heuristic_id is not None
        assert (tmp_path / "claude" / f"{heuristic_id}.json").exists()

    def test_take_profit_creates_heuristic_file(self, tmp_path):
        portfolio = get_portfolio("gpt")
        _open_position(portfolio, ticker="MSFT")

        closed_trades = portfolio.update_prices({"MSFT": 120.0})
        assert len(closed_trades) == 1
        assert closed_trades[0].exit_reason == "take_profit"

        with patch("src.agent.erl._call_model", return_value=_GOOD_ERL_RESPONSE):
            heuristic_id = run_erl(
                track="gpt",
                trade=_build_erl_trade_dict(closed_trades[0]),
                technicals_str=closed_trades[0].technical_snapshot,
                regime_str=closed_trades[0].regime,
            )

        assert heuristic_id is not None
        assert (tmp_path / "gpt" / f"{heuristic_id}.json").exists()

    def test_heuristic_file_content(self, tmp_path):
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 90.0})[0]

        with patch("src.agent.erl._call_model", return_value=_GOOD_ERL_RESPONSE):
            heuristic_id = run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )

        data = json.loads((tmp_path / "claude" / f"{heuristic_id}.json").read_text())
        assert data["track"] == "claude"
        assert data["quality_score"] == 7.0
        assert data["market"] == "us"
        assert data["regime"] == "trending"
        assert "RSI" in data["trigger"]
        assert "long" in data["action"].lower()
        assert data["source_trade_id"] == closed.trade_id
        assert data["access_count"] == 0
        assert data["is_core"] is False

    def test_two_trades_two_heuristic_files(self, tmp_path):
        portfolio = get_portfolio("claude")
        for ticker in ("AAPL", "MSFT"):
            _open_position(portfolio, ticker=ticker)
            closed = portfolio.update_prices({ticker: 90.0})[0]
            with patch("src.agent.erl._call_model", return_value=_GOOD_ERL_RESPONSE):
                run_erl(
                    track="claude",
                    trade=_build_erl_trade_dict(closed),
                    technicals_str=closed.technical_snapshot,
                    regime_str=closed.regime,
                )

        files = list((tmp_path / "claude").glob("*.json"))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# Technical snapshot flows through the full lifecycle
# ---------------------------------------------------------------------------

class TestTechnicalSnapshotLifecycle:
    def test_snapshot_stored_in_open_position(self):
        portfolio = get_portfolio("claude")
        snapshot = "RSI=61.4, EMA20=above, BB_width=0.07, ATR=2.1"
        pos = _open_position(portfolio, snapshot=snapshot)
        assert pos is not None
        assert pos.technical_snapshot == snapshot

    def test_snapshot_preserved_in_closed_trade_on_stop(self):
        snapshot = "RSI=44.0, EMA20=below, PSAR=above_price"
        portfolio = get_portfolio("claude")
        _open_position(portfolio, snapshot=snapshot)
        closed = portfolio.update_prices({"AAPL": 90.0})[0]
        assert closed.technical_snapshot == snapshot

    def test_snapshot_preserved_in_closed_trade_on_target(self):
        snapshot = "RSI=58.3, EMA20=above, volume_spike=True"
        portfolio = get_portfolio("claude")
        _open_position(portfolio, snapshot=snapshot)
        closed = portfolio.update_prices({"AAPL": 120.0})[0]
        assert closed.technical_snapshot == snapshot

    def test_regime_preserved_in_closed_trade(self):
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 90.0})[0]
        assert closed.regime == "trending"

    def test_confidence_preserved_in_closed_trade(self):
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 90.0})[0]
        assert closed.confidence == pytest.approx(0.85, rel=1e-6)


# ---------------------------------------------------------------------------
# MIPRO entry-input capture flows through the full lifecycle
# ---------------------------------------------------------------------------

_ENTRY_INPUTS = {
    "technicals": "RSI=55.2, EMA20=above, ATR=1.5",
    "regime": "trending; recommended tactic: EMA crossover entries",
    "news_summary": "Neutral news. Insider activity: none",
    "macro_context": "No macro data available.",
    "heuristics": "No relevant heuristics yet.",
}


def _open_with_inputs(portfolio, ticker: str = "AAPL"):
    return portfolio.open_trade(
        ticker=ticker,
        market="us",
        quantity=10.0,
        entry_price=100.0,
        stop_loss=95.0,
        target=115.0,
        regime="trending",
        reasoning="Strong uptrend continuation",
        confidence=0.85,
        technical_snapshot="RSI=55.2",
        sector="Technology",
        entry_inputs=dict(_ENTRY_INPUTS),
    )


class TestEntryInputsLifecycle:
    def test_entry_inputs_stored_on_open_position(self):
        portfolio = get_portfolio("claude")
        pos = _open_with_inputs(portfolio)
        assert pos.entry_inputs == _ENTRY_INPUTS

    def test_entry_inputs_preserved_in_closed_trade(self):
        portfolio = get_portfolio("claude")
        _open_with_inputs(portfolio)
        closed = portfolio.update_prices({"AAPL": 120.0})[0]
        assert closed.entry_inputs == _ENTRY_INPUTS

    def test_entry_inputs_default_empty_when_not_provided(self):
        portfolio = get_portfolio("claude")
        pos = _open_position(portfolio)  # no entry_inputs passed
        assert pos.entry_inputs == {}

    def test_optimizer_filter_accepts_captured_trades(self):
        """The old hasattr(_entry_inputs) check always failed — verify trades now qualify."""
        portfolio = get_portfolio("claude")
        for i in range(3):
            _open_with_inputs(portfolio, ticker=f"TICK{i}")
            portfolio.update_prices({f"TICK{i}": 120.0})  # close at target

        # Mirror optimizer.py's filter + labelling exactly
        qualifying = [t for t in portfolio.closed_trades if getattr(t, "entry_inputs", None)]
        assert len(qualifying) == 3
        assert qualifying[0].entry_inputs["technicals"] == _ENTRY_INPUTS["technicals"]
        # Winners label as BUY, losers as HOLD
        assert ["BUY" if t.pnl_pct > 0 else "HOLD" for t in qualifying] == ["BUY", "BUY", "BUY"]


# ---------------------------------------------------------------------------
# Exit fill model — stops/targets fill at the level, not the raw gapped price
# ---------------------------------------------------------------------------

class TestExitFillModel:
    def test_take_profit_caps_at_target_not_gap(self):
        # entry 100, target 115; a gap to 200 must still fill at ~target, not +100%.
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 200.0})[0]
        assert closed.exit_reason == "take_profit"
        assert closed.exit_price <= 115.0
        assert closed.pnl_pct == pytest.approx(0.149, abs=0.01)

    def test_stop_loss_gap_is_bounded(self):
        # entry 100, stop 95; a crash to 50 must not book a ~-50% loss.
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 50.0})[0]
        assert closed.exit_reason == "stop_loss"
        floor = 95.0 * (1 - settings.max_gap_slippage_pct) * (1 - settings.simulated_slippage)
        assert closed.exit_price >= floor
        assert closed.pnl_pct > -0.10

    def test_stop_loss_without_gap_fills_at_stop(self):
        # Price lands exactly on the stop → no gap slippage, fill ≈ stop.
        portfolio = get_portfolio("claude")
        _open_position(portfolio)
        closed = portfolio.update_prices({"AAPL": 95.0})[0]
        assert closed.exit_reason == "stop_loss"
        assert closed.exit_price == pytest.approx(95.0 * (1 - settings.simulated_slippage))


# ---------------------------------------------------------------------------
# ERL filter logic (low quality / unparseable)
# ---------------------------------------------------------------------------

class TestErlFilter:
    def _close_trade(self, track: str = "claude"):
        portfolio = get_portfolio(track)
        _open_position(portfolio)
        return portfolio.update_prices({"AAPL": 90.0})[0]

    def test_low_quality_response_returns_none(self):
        closed = self._close_trade()
        with patch("src.agent.erl._call_model", return_value=_LOW_QUALITY_ERL_RESPONSE):
            result = run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )
        assert result is None

    def test_low_quality_response_writes_no_file(self, tmp_path):
        closed = self._close_trade()
        with patch("src.agent.erl._call_model", return_value=_LOW_QUALITY_ERL_RESPONSE):
            run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )
        claude_dir = tmp_path / "claude"
        assert not list(claude_dir.glob("*.json")) if claude_dir.exists() else True

    def test_unparseable_response_returns_none(self):
        closed = self._close_trade()
        with patch("src.agent.erl._call_model", return_value=_UNPARSEABLE_ERL_RESPONSE):
            result = run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )
        assert result is None

    def test_none_model_response_returns_none(self):
        closed = self._close_trade()
        with patch("src.agent.erl._call_model", return_value=None):
            result = run_erl(
                track="claude",
                trade=_build_erl_trade_dict(closed),
                technicals_str=closed.technical_snapshot,
                regime_str=closed.regime,
            )
        assert result is None


# ---------------------------------------------------------------------------
# HeuristicStore: save / retrieve / access-count
# ---------------------------------------------------------------------------

class TestHeuristicStoreBehavior:
    def test_save_creates_file(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        heuristic_id = store.save(
            trigger="Price above EMA50",
            action="Consider entering long",
            market="us",
            regime="trending",
            quality_score=6.0,
        )
        assert (tmp_path / "claude" / f"{heuristic_id}.json").exists()

    def test_retrieve_increments_access_count(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        heuristic_id = store.save(
            trigger="RSI below 30", action="Wait for reversal",
            market="us", regime="any", quality_score=5.0,
        )

        store.retrieve(ticker="AAPL", regime="any", market="us", top_k=1)
        store.retrieve(ticker="AAPL", regime="any", market="us", top_k=1)

        data = json.loads((tmp_path / "claude" / f"{heuristic_id}.json").read_text())
        assert data["access_count"] == 2

    def test_prune_removes_low_quality_low_access(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        keep_id = store.save(trigger="A", action="B", quality_score=6.0)
        prune_id = store.save(trigger="C", action="D", quality_score=2.0)

        removed = store.prune(quality_threshold=4.0, access_threshold=2)

        assert removed == 1
        assert (tmp_path / "claude" / f"{keep_id}.json").exists()
        assert not (tmp_path / "claude" / f"{prune_id}.json").exists()

    def test_promote_core_marks_frequently_accessed(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        heuristic_id = store.save(trigger="A", action="B", quality_score=7.0)

        # Retrieve 10 times to hit the promotion threshold
        for _ in range(10):
            store.retrieve(ticker="ANY", regime="any", market="us", top_k=1)

        store.promote_core(access_threshold=10)
        data = json.loads((tmp_path / "claude" / f"{heuristic_id}.json").read_text())
        assert data["is_core"] is True
