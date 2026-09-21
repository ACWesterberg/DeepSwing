from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from config.settings import settings
from src.agent.compiled_program import save_compiled_program
from src.agent.outcomes import evaluate_forward_path
from src.agent.replay import ReplayExample, dspy_program, load_corpus, oracle, score_program
from src.agent.risk import validate_trade
from src.portfolio.daily import DailyPortfolio
from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import Portfolio
from tests.test_risk import _make_signals


def _example() -> ReplayExample:
    return ReplayExample("gpt", "TEST", "us", "2026-08-01", {
        "technicals": "t", "regime": "r", "news_summary": "n", "macro_context": "m",
        "heuristics": "h", "heuristic_ids": ["internal-only"],
    }, "PASS", -1.0)


def _bars(*rows: tuple[float, float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2026-08-02", periods=len(rows)))


def _open(portfolio: Portfolio):
    return portfolio.open_trade("TEST", "us", 10, 100, 97, 120, "trending", "test", 0.8)


class TestEvaluationFailures:
    def test_provider_failure_cannot_earn_pass_credit(self):
        program = Mock(side_effect=RuntimeError("provider unavailable"))
        with pytest.raises(RuntimeError, match="Replay failed"):
            score_program([_example()], dspy_program(program))

    def test_metadata_is_not_sent_to_the_program(self):
        program = Mock(return_value=SimpleNamespace(action="PASS"))
        assert score_program([_example()], dspy_program(program)).n == 1
        assert "heuristic_ids" not in program.call_args.kwargs
        assert len(program.call_args.kwargs) == 5

    def test_inconsistent_oracle_labels_abort(self):
        example = _example()
        example.r_multiple = 0.1
        assert oracle(example) == "BUY"
        with pytest.raises(ValueError, match="Invalid outcome"):
            score_program([example], oracle)

    def test_legacy_gross_corpus_requires_rebuild(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text("[]")
        with pytest.raises(ValueError, match="rebuild"):
            load_corpus(path)


class TestArtifactReplacement:
    def test_failed_save_keeps_the_incumbent(self, tmp_path):
        path = tmp_path / "gpt_trade_decision.json"
        path.write_text('{"instructions": "old"}')
        program = Mock()
        program.save.side_effect = RuntimeError("disk failure")
        with pytest.raises(RuntimeError):
            save_compiled_program(program, path, Mock())
        assert json.loads(path.read_text())["instructions"] == "old"
        assert len(list(tmp_path.iterdir())) == 1

    def test_failed_validation_keeps_the_incumbent(self, tmp_path):
        path = tmp_path / "gpt_trade_decision.json"
        path.write_text('{"instructions": "old"}')
        program = Mock()
        from pathlib import Path
        program.save.side_effect = lambda p: Path(p).write_text('{"instructions": "new"}')
        with pytest.raises(ValueError):
            save_compiled_program(program, path, Mock(side_effect=ValueError("invalid schema")))
        assert json.loads(path.read_text())["instructions"] == "old"
        assert len(list(tmp_path.iterdir())) == 1

    def test_validated_program_replaces_and_archives(self, tmp_path):
        from pathlib import Path
        path = tmp_path / "gpt_trade_decision.json"
        path.write_text('{"instructions": "old"}')
        program = Mock()
        program.save.side_effect = lambda p: Path(p).write_text('{"instructions": "new"}')
        validate = Mock()
        save_compiled_program(program, path, validate)
        validate.assert_called_once()
        assert json.loads(path.read_text())["instructions"] == "new"
        archive, = tmp_path.glob("gpt_trade_decision_*.json")
        assert json.loads(archive.read_text())["instructions"] == "old"


class TestRiskCorrections:
    @pytest.mark.parametrize("cash", [100_000.0, 5_000.0])
    def test_drawdown_halves_the_final_capped_size(self, cash):
        args = dict(action="BUY", entry_price=100, stop_loss=97, target=110,
                    portfolio_equity=100_000, open_positions=[], signals=_make_signals(),
                    available_cash=cash, market="us")
        normal = validate_trade(**args)
        reduced = validate_trade(**args, is_drawdown_mode=True)
        assert normal.approved and reduced.approved
        assert reduced.quantity == pytest.approx(normal.quantity / 2)
        assert reduced.risk_amount == pytest.approx(normal.risk_amount / 2)

    @pytest.mark.parametrize("field", ["entry_price", "stop_loss", "target", "portfolio_equity", "available_cash"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_bad_numbers_cannot_pass_risk_validation(self, field, bad):
        args = dict(action="BUY", entry_price=100, stop_loss=97, target=110,
                    portfolio_equity=100_000, open_positions=[], signals=_make_signals(),
                    available_cash=100_000)
        assert not validate_trade(**{**args, field: bad}).approved

    def test_bad_quote_does_not_corrupt_portfolio_state(self):
        portfolio = Portfolio("gpt")
        position = _open(portfolio)
        equity = portfolio.equity
        portfolio.update_prices({"TEST": float("nan")})
        assert portfolio.equity == equity
        assert portfolio.close_trade(position.trade_id, float("inf"), "stop_loss") is None
        assert len(portfolio.open_positions) == 1


class TestOutcomeParity:
    def test_gap_below_stop_includes_gap_loss_and_both_fee_legs(self):
        outcome = evaluate_forward_path(_bars((90, 92, 89, 91)), 100, 2, "us")
        portfolio = Portfolio("gpt")
        position = portfolio.open_trade("TEST", "us", 1, 100, 97, 107.5, "trending", "", 0.5)
        closed = portfolio.close_trade(position.trade_id, 90, "stop_loss")
        assert outcome.r_multiple == pytest.approx(closed.rrr_achieved)
        assert outcome.pnl_pct == pytest.approx(closed.pnl_pct)
        assert outcome.r_multiple < -3.33
        assert outcome.action == "PASS"

    def test_breakeven_is_net_zero_and_not_a_positive_pass_label(self):
        outcome = evaluate_forward_path(_bars((100, 104, 99, 103), (103, 104, 99, 101)), 100, 2, "us")
        assert outcome.exit_reason == "breakeven_stop"
        assert outcome.r_multiple == 0.0
        assert outcome.action == "PASS"

    def test_trailing_stop_is_applied_to_counterfactuals(self):
        outcome = evaluate_forward_path(_bars((100, 107, 99, 107), (106, 107, 102, 104)), 100, 2, "us")
        assert outcome.exit_reason == "trailing_stop"
        assert outcome.action == "BUY"
        assert 0 < outcome.r_multiple < settings.min_rrr

    def test_fx_costs_affect_the_same_underlying_path(self):
        window = _bars((100, 101, 99, 100.1))
        us = evaluate_forward_path(window, 100, 2, "us")
        nordic = evaluate_forward_path(window, 100, 2, "nordic")
        assert us.r_multiple < nordic.r_multiple < 0

    def test_gap_above_target_exits_before_a_later_low(self):
        outcome = evaluate_forward_path(_bars((110, 112, 90, 100)), 100, 2, "us")
        assert outcome.exit_reason == "take_profit"
        assert outcome.r_multiple > 2.5

    def test_corrupt_bars_fail_evaluation(self):
        with pytest.raises(ValueError, match="Invalid OHLC"):
            evaluate_forward_path(_bars((100, float("nan"), 99, 101)), 100, 2, "us")


class TestAccounting:
    def test_cost_only_loser_is_not_counted_as_a_win(self):
        from src.backtesting.engine import _compute_metrics
        portfolio = DailyPortfolio(100_000, commission_rate=0.002)
        portfolio.open_position("TEST", 100, 97, 110, 10, date(2026, 8, 1))
        portfolio.close_position("TEST", 100.1, date(2026, 8, 2), "end_of_window")
        result = _compute_metrics(portfolio.closed_trades, 100_000)
        assert result["total_pnl"] == pytest.approx(-3.0)
        assert result["win_rate"] == 0
        assert result["avg_rrr"] < 0
        assert result["optimization_metric"] < 0

    def test_open_losses_are_reported_before_first_close(self):
        portfolio = Portfolio("gpt")
        position = _open(portfolio)
        position.current_price = 90
        metrics = compute_metrics(portfolio)
        assert metrics.total_return_pct == pytest.approx((portfolio.equity / portfolio.starting_equity - 1) * 100)
        assert metrics.max_drawdown_pct > 0


class TestRuleMeaning:
    @pytest.mark.parametrize("left,right", [
        ("buy breakout", "do not buy breakout"),
        ("buy breakout", "don't buy breakout"),
        ("buy when RSI above 70", "buy when RSI below 70"),
        ("buy when RSI > 70", "buy when RSI < 70"),
    ])
    def test_opposite_rules_are_not_duplicates(self, left, right):
        from src.agent.memory import similarity
        assert similarity(left, right) == 0


def test_backtest_entry_cannot_see_an_earlier_low(monkeypatch):
    import src.backtesting.engine as engine
    from src.analysis.screener import ScreenerCandidate
    from src.analysis.regime import RegimeResult

    window = _bars((100, 101, 99, 100), (99, 101, 96, 100), (100, 101, 99, 100))
    signals = _make_signals()
    regime = RegimeResult("trending", 0.6, 0.1, "test")
    monkeypatch.setattr(engine, "_WARMUP_DAYS", 2)
    monkeypatch.setattr(engine, "compute_signals", lambda *args: signals)
    monkeypatch.setattr(engine, "classify_regime", lambda *args: regime)
    monkeypatch.setattr(engine, "screen_candidates", lambda *args: [ScreenerCandidate("TEST", "us", signals, regime)])
    backtest = engine.BacktestEngine("us", ["TEST"], date(2026, 8, 3), date(2026, 8, 4))
    result = backtest._simulate_window({"TEST": window}, 0, date(2026, 8, 3), date(2026, 8, 4))
    trade, = result.trades
    assert trade.entry_date == date(2026, 8, 3)
    assert trade.exit_date == date(2026, 8, 4)
    assert trade.exit_reason == "end_of_window"


def test_mechanical_stop_runs_before_news_review(monkeypatch):
    import src.scheduler.scan_loop as scan
    portfolio = Portfolio("gpt")
    _open(portfolio)
    monkeypatch.setattr(settings, "tracks", ["gpt"])
    monkeypatch.setattr(scan, "get_portfolio", lambda track: portfolio)
    monkeypatch.setattr(scan, "_get_current_prices", lambda *args: {"TEST": 90})
    review = Mock()
    monkeypatch.setattr(scan, "_maybe_news_exit", review)
    monkeypatch.setattr(scan, "_emit_close", Mock())
    monkeypatch.setattr(scan, "_persist_decisions", Mock())
    monkeypatch.setattr(scan, "persist_portfolio", Mock())
    scan._monitor_holdings("us")
    review.assert_not_called()
    assert portfolio.closed_trades[0].exit_reason == "stop_loss"
