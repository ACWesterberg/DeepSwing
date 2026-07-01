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
