from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scheduler.optimizer import _pnl_weighted_metric


def _pred(action: str) -> SimpleNamespace:
    return SimpleNamespace(action=action)


def _example(r_multiple: float) -> SimpleNamespace:
    return SimpleNamespace(r_multiple=r_multiple)


class TestPnlWeightedMetric:
    def test_passing_is_neutral_regardless_of_outcome(self):
        assert _pnl_weighted_metric(_example(2.0), _pred("PASS")) == pytest.approx(0.5)
        assert _pnl_weighted_metric(_example(-2.0), _pred("PASS")) == pytest.approx(0.5)

    def test_taking_a_winner_beats_passing(self):
        assert _pnl_weighted_metric(_example(1.0), _pred("BUY")) > 0.5

    def test_taking_a_loser_scores_below_passing(self):
        assert _pnl_weighted_metric(_example(-1.0), _pred("BUY")) < 0.5

    def test_right_tail_stays_visible(self):
        """The re-tune targets bigger R. A metric that saturates by +2R would
        rank a 5R trade as barely better than a 2.5R one and quietly undo it."""
        good = _pnl_weighted_metric(_example(2.5), _pred("BUY"))
        great = _pnl_weighted_metric(_example(5.0), _pred("BUY"))
        assert great - good > 0.05

    def test_bigger_winner_scores_higher(self):
        small = _pnl_weighted_metric(_example(1.0), _pred("BUY"))
        big = _pnl_weighted_metric(_example(3.0), _pred("BUY"))
        assert big > small

    def test_passing_a_loser_beats_taking_it(self):
        """Correctly avoiding a losing setup should out-score entering it."""
        avoided = _pnl_weighted_metric(_example(-1.0), _pred("PASS"))
        taken = _pnl_weighted_metric(_example(-1.0), _pred("BUY"))
        assert avoided > taken

    def test_output_bounded_between_zero_and_one(self):
        for r in (-50.0, -1.0, 0.0, 2.5, 50.0):
            for action in ("BUY", "PASS"):
                score = _pnl_weighted_metric(_example(r), _pred(action))
                assert 0.0 <= score <= 1.0

    def test_missing_r_defaults_to_neutral(self):
        assert _pnl_weighted_metric(SimpleNamespace(), _pred("BUY")) == pytest.approx(0.5)

    def test_missing_action_treated_as_pass(self):
        assert _pnl_weighted_metric(_example(2.0), SimpleNamespace()) == pytest.approx(0.5)

    def test_case_insensitive_action(self):
        assert _pnl_weighted_metric(_example(1.0), _pred("buy")) > 0.5


class TestProgramFingerprint:
    """A compiled program is applied the moment MIPRO writes it. Without an
    identity stamped on its decisions, no compile can ever be evaluated."""

    def test_identical_content_hashes_the_same(self, tmp_path):
        from src.agent.compiled_program import program_fingerprint

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text('{"instructions": "Buy strength.", "demos": []}')
        b.write_text('{"demos": [], "instructions": "Buy strength."}')  # key order differs
        assert program_fingerprint(a) == program_fingerprint(b)
        assert len(program_fingerprint(a)) == 12

    def test_changed_instructions_hash_differently(self, tmp_path):
        from src.agent.compiled_program import program_fingerprint

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text('{"instructions": "Buy strength."}')
        b.write_text('{"instructions": "Buy weakness."}')
        assert program_fingerprint(a) != program_fingerprint(b)

    def test_unreadable_program_is_not_fatal(self, tmp_path):
        from src.agent.compiled_program import program_fingerprint

        bad = tmp_path / "bad.json"
        bad.write_text("not json")
        assert program_fingerprint(bad) is None
        assert program_fingerprint(tmp_path / "missing.json") is None


class TestMetricsByProgram:
    def _trade(self, pnl_pct, program_hash=""):
        return SimpleNamespace(pnl_pct=pnl_pct, program_hash=program_hash)

    def test_groups_trades_by_the_program_that_decided_them(self):
        from src.portfolio.metrics import metrics_by_program

        portfolio = SimpleNamespace(closed_trades=[
            self._trade(0.10, "aaa111"), self._trade(0.06, "aaa111"),
            self._trade(-0.04, "bbb222"),
            self._trade(0.02),  # uncompiled baseline
        ])
        rows = {r["program"]: r for r in metrics_by_program(portfolio)}
        assert rows["aaa111"]["trades"] == 2
        assert rows["aaa111"]["win_rate"] == pytest.approx(1.0)
        assert rows["aaa111"]["mean_pnl_pct"] == pytest.approx(0.08)
        assert rows["bbb222"]["mean_pnl_pct"] == pytest.approx(-0.04)
        assert rows["baseline"]["trades"] == 1

    def test_trades_predating_fingerprinting_group_as_baseline(self):
        from src.portfolio.metrics import metrics_by_program

        legacy = SimpleNamespace(pnl_pct=0.05)  # no program_hash attribute at all
        assert metrics_by_program(SimpleNamespace(closed_trades=[legacy]))[0]["program"] == "baseline"

    def test_empty_portfolio(self):
        from src.portfolio.metrics import metrics_by_program

        assert metrics_by_program(SimpleNamespace(closed_trades=[])) == []


class TestForwardSplit:
    """The shuffle balanced the slices but let a validation example predate a
    training one; splitting each group at its own boundary gives both."""

    def test_each_group_is_split_on_its_own_time_boundary(self):
        from src.scheduler.optimizer import _forward_split

        real = [f"real{i}" for i in range(10)]          # chronological
        counterfactual = [f"cf{i}" for i in range(10)]
        train, val = _forward_split(real, counterfactual, val_frac=0.2)

        # Both kinds appear in both slices — the problem the shuffle solved.
        assert any(e.startswith("real") for e in val)
        assert any(e.startswith("cf") for e in val)
        # And validation is strictly later than training within each kind.
        assert val == ["real8", "real9", "cf8", "cf9"]

    def test_never_returns_an_empty_validation_set(self):
        from src.scheduler.optimizer import _forward_split

        train, val = _forward_split(["only"])
        assert val and train

    def test_empty_groups_are_skipped(self):
        from src.scheduler.optimizer import _forward_split

        train, val = _forward_split([f"a{i}" for i in range(5)], [])
        assert len(train) + len(val) == 5


class TestOptimizationGates:
    def test_gate_leaves_a_usable_validation_set(self):
        """At the old floor of 10, MIPRO's 20% holdout was two examples."""
        from src.scheduler.optimizer import MIN_EXAMPLES_FOR_OPTIMIZATION

        assert MIN_EXAMPLES_FOR_OPTIMIZATION >= 25
        assert int(MIN_EXAMPLES_FOR_OPTIMIZATION * 0.2) >= 5

    def test_a_real_example_floor_survives_alongside_the_total(self):
        """Counterfactuals top the trainset up, but must not constitute it."""
        from src.scheduler.optimizer import (
            MIN_EXAMPLES_FOR_OPTIMIZATION,
            MIN_REAL_EXAMPLES,
        )
        assert 0 < MIN_REAL_EXAMPLES < MIN_EXAMPLES_FOR_OPTIMIZATION


class TestOptunaGuard:
    """MIPROv2 imports optuna deep inside compile(), after demo bootstrapping
    and instruction proposal have already spent minutes of heavy prompt-model
    calls. A live weekly run got that far and died there, weekly, silently."""

    def test_missing_optuna_bails_with_that_reason(self, monkeypatch, caplog):
        import builtins
        import logging
        import src.scheduler.optimizer as opt

        real_import = builtins.__import__

        def _no_optuna(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("MIPROv2 requires optional dependency 'optuna'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_optuna)
        portfolio = SimpleNamespace(closed_trades=[SimpleNamespace()] * 40)
        monkeypatch.setattr(opt, "get_portfolio", lambda track: portfolio)

        with caplog.at_level(logging.ERROR, logger="src.scheduler.optimizer"):
            assert opt.run_mipro_optimization("gpt") is False

        # Must bail for THIS reason — a run that falls through to an empty
        # trainset also returns False, which is what made an earlier version of
        # this test pass with the guard removed.
        assert any("optuna" in r.message for r in caplog.records)

    def test_run_proceeds_past_the_guard_when_optuna_is_present(self, monkeypatch, caplog):
        import logging
        pytest.importorskip("optuna")
        import src.scheduler.optimizer as opt

        portfolio = SimpleNamespace(closed_trades=[SimpleNamespace()] * 40)
        monkeypatch.setattr(opt, "get_portfolio", lambda track: portfolio)

        with caplog.at_level(logging.INFO, logger="src.scheduler.optimizer"):
            opt.run_mipro_optimization("gpt")

        assert any("starting optimization" in r.message for r in caplog.records)

    def test_optuna_is_declared_as_a_dependency(self):
        from pathlib import Path
        reqs = Path(__file__).resolve().parent.parent / "requirements.txt"
        assert "optuna" in reqs.read_text()


class TestCounterfactualCap:
    """Counterfactuals used to be capped at parity with real trades, tying the
    trainset to the one input that takes months to accumulate."""

    def test_cap_scales_with_trainset_rather_than_matching_it(self):
        from src.scheduler.optimizer import counterfactual_cap
        from config.settings import settings

        assert settings.counterfactual_ratio_cap > 1.0
        # At the MIPRO threshold, the trainset should be several times the
        # number of real trades, not double it.
        assert counterfactual_cap(30) > 30
        assert counterfactual_cap(30) == int(30 * settings.counterfactual_ratio_cap)

    def test_cap_is_monotonic_in_real_examples(self):
        from src.scheduler.optimizer import counterfactual_cap
        assert counterfactual_cap(10) < counterfactual_cap(20) < counterfactual_cap(40)

    def test_absolute_ceiling_still_applies(self):
        from src.scheduler.optimizer import counterfactual_cap
        from config.settings import settings
        assert counterfactual_cap(10_000) == settings.counterfactual_max_examples

    def test_no_real_examples_means_no_counterfactuals(self):
        from src.scheduler.optimizer import counterfactual_cap
        assert counterfactual_cap(0) == 0

    def test_validation_split_is_large_enough_to_select_on(self):
        from src.scheduler.optimizer import counterfactual_cap, MIN_TRADES_FOR_OPTIMIZATION
        total = MIN_TRADES_FOR_OPTIMIZATION + counterfactual_cap(MIN_TRADES_FOR_OPTIMIZATION)
        # MIPRO holds out 20% and picks the winning instructions on that slice.
        # Parity gave 12 examples, which is noise.
        assert int(total * 0.2) >= 25, f"val split of {int(total*0.2)} is too small to select on"
