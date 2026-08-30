from __future__ import annotations

import json

import pytest

from src.agent.replay import (
    ReplayExample,
    always_buy,
    always_pass,
    load_corpus,
    oracle,
    save_corpus,
    score_program,
)


def _ex(label: str, r: float, ticker: str = "AAPL") -> ReplayExample:
    return ReplayExample(
        track="claude", ticker=ticker, market="us", timestamp="2026-08-01T09:00:00",
        entry_inputs={"technicals": "t", "regime": "r", "news_summary": "n",
                      "macro_context": "m", "heuristics": "h"},
        label=label, r_multiple=r,
    )


# A corpus with genuine edge available: the winners more than pay for the losers.
_PROFITABLE = [_ex("BUY", 3.0), _ex("BUY", 2.5), _ex("PASS", -1.0),
               _ex("PASS", -1.0), _ex("PASS", -1.0)]

# A corpus with no edge: taking everything loses.
_LOSING = [_ex("BUY", 2.5), _ex("PASS", -1.0), _ex("PASS", -1.0),
           _ex("PASS", -1.0), _ex("PASS", -1.0)]


class TestHarnessCanDistinguishPrograms:
    """If the harness can't order these three, it measures nothing and no
    verdict it gives about a real prompt should be believed."""

    def test_oracle_beats_always_buy_beats_always_pass(self):
        o = score_program(_PROFITABLE, oracle, "oracle")
        b = score_program(_PROFITABLE, always_buy, "always_buy")
        p = score_program(_PROFITABLE, always_pass, "always_pass")
        assert o.mean_metric > b.mean_metric > p.mean_metric

    def test_passing_is_correct_when_there_is_no_edge(self):
        # The metric's PASS floor is exactly 0.5, so on a losing corpus the
        # do-nothing program should win. This is a real property of the live
        # objective, not a harness artefact — worth pinning so it stays visible.
        b = score_program(_LOSING, always_buy, "always_buy")
        p = score_program(_LOSING, always_pass, "always_pass")
        assert p.mean_metric > b.mean_metric
        assert p.mean_metric == pytest.approx(0.5)

    def test_always_pass_scores_exactly_half_on_any_corpus(self):
        for corpus in (_PROFITABLE, _LOSING):
            assert score_program(corpus, always_pass).mean_metric == pytest.approx(0.5)


class TestRealisedROutrunsTheMetric:
    """The metric alone hides the inert program; the R columns are what expose it."""

    def test_do_nothing_captures_no_r_despite_a_respectable_metric(self):
        p = score_program(_PROFITABLE, always_pass)
        assert p.mean_metric == pytest.approx(0.5)   # looks unremarkable, not bad
        assert p.buys == 0
        assert p.total_r_taken == 0.0
        assert p.recall == 0.0                        # this is what gives it away

    def test_oracle_captures_every_paying_setup(self):
        o = score_program(_PROFITABLE, oracle)
        assert o.buys == 2
        assert o.total_r_taken == pytest.approx(5.5)
        assert o.recall == 1.0
        assert o.precision == 1.0
        assert o.false_buys == 0

    def test_indiscriminate_buying_is_penalised_on_precision(self):
        b = score_program(_PROFITABLE, always_buy)
        assert b.recall == 1.0            # it did take every winner
        assert b.precision == pytest.approx(2 / 5)   # and every loser too
        assert b.false_buys == 3


class TestScoringMechanics:
    def test_unknown_action_is_treated_as_pass(self):
        r = score_program(_PROFITABLE, lambda e: "HOLD")
        assert r.buys == 0
        assert r.mean_metric == pytest.approx(0.5)

    def test_empty_corpus_is_an_error_not_a_score(self):
        with pytest.raises(ValueError):
            score_program([], always_buy)

    def test_missed_buys_counted(self):
        r = score_program(_PROFITABLE, always_pass)
        assert r.missed_buys == 2


class TestCorpusRoundTrip:
    """The corpus is cached so comparing programs costs no further price data."""

    def test_save_and_load_preserves_examples(self, tmp_path):
        path = tmp_path / "corpus.json"
        assert save_corpus(_PROFITABLE, path) == len(_PROFITABLE)

        loaded = load_corpus(path)
        assert len(loaded) == len(_PROFITABLE)
        assert loaded[0].label == _PROFITABLE[0].label
        assert loaded[0].r_multiple == pytest.approx(_PROFITABLE[0].r_multiple)
        assert loaded[0].entry_inputs == _PROFITABLE[0].entry_inputs

    def test_loaded_corpus_scores_identically(self, tmp_path):
        path = tmp_path / "corpus.json"
        save_corpus(_PROFITABLE, path)
        assert score_program(load_corpus(path), oracle).mean_metric == pytest.approx(
            score_program(_PROFITABLE, oracle).mean_metric
        )

    def test_cache_is_plain_json(self, tmp_path):
        path = tmp_path / "corpus.json"
        save_corpus(_PROFITABLE, path)
        rows = json.loads(path.read_text())
        assert rows[0]["ticker"] == "AAPL"
        assert "entry_inputs" in rows[0]
