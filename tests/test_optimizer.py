from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.scheduler.optimizer import _pnl_weighted_metric


def _pred(action: str) -> SimpleNamespace:
    return SimpleNamespace(action=action)


def _example(pnl_pct: float) -> SimpleNamespace:
    return SimpleNamespace(pnl_pct=pnl_pct)


class TestPnlWeightedMetric:
    def test_passing_is_neutral_regardless_of_outcome(self):
        assert _pnl_weighted_metric(_example(0.20), _pred("PASS")) == pytest.approx(0.5)
        assert _pnl_weighted_metric(_example(-0.20), _pred("PASS")) == pytest.approx(0.5)

    def test_taking_a_winner_beats_passing(self):
        assert _pnl_weighted_metric(_example(0.05), _pred("BUY")) > 0.5

    def test_taking_a_loser_scores_below_passing(self):
        assert _pnl_weighted_metric(_example(-0.05), _pred("BUY")) < 0.5

    def test_bigger_winner_scores_higher(self):
        small = _pnl_weighted_metric(_example(0.03), _pred("BUY"))
        big = _pnl_weighted_metric(_example(0.15), _pred("BUY"))
        assert big > small

    def test_passing_a_loser_beats_taking_it(self):
        """Correctly avoiding a losing setup should out-score entering it."""
        avoided = _pnl_weighted_metric(_example(-0.10), _pred("PASS"))
        taken = _pnl_weighted_metric(_example(-0.10), _pred("BUY"))
        assert avoided > taken

    def test_output_bounded_between_zero_and_one(self):
        for pnl in (-5.0, -0.5, 0.0, 0.5, 5.0):
            for action in ("BUY", "PASS"):
                score = _pnl_weighted_metric(_example(pnl), _pred(action))
                assert 0.0 <= score <= 1.0

    def test_missing_pnl_defaults_to_neutral(self):
        assert _pnl_weighted_metric(SimpleNamespace(), _pred("BUY")) == pytest.approx(0.5)

    def test_missing_action_treated_as_pass(self):
        assert _pnl_weighted_metric(_example(0.10), SimpleNamespace()) == pytest.approx(0.5)

    def test_case_insensitive_action(self):
        assert _pnl_weighted_metric(_example(0.05), _pred("buy")) > 0.5


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
