from __future__ import annotations

import json

import pytest

from src.agent.replay import (
    _batch_prices,
    headroom,
    select_decision_rows,
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

    def test_passing_now_tracks_the_loss_it_avoided(self):
        # A PASS earns the R it avoided, so there is no fixed 0.5 baseline any
        # more. On a corpus of losers, declining them scores above neutral.
        assert score_program(_LOSING, always_pass).mean_metric > 0.5
        # And on one where most setups paid, declining them scores below it.
        rich = [_ex("BUY", 3.0), _ex("BUY", 2.5), _ex("BUY", 2.0), _ex("PASS", -1.0)]
        assert score_program(rich, always_pass).mean_metric < 0.5

    def test_avoiding_a_loser_scores_like_catching_a_winner(self):
        caught = score_program([_ex("BUY", 1.0)], always_buy).mean_metric
        avoided = score_program([_ex("PASS", -1.0)], always_pass).mean_metric
        assert caught == pytest.approx(avoided)


class TestRealisedROutrunsTheMetric:
    """The metric alone hides the inert program; the R columns are what expose it."""

    def test_do_nothing_captures_no_r_despite_a_respectable_metric(self):
        p = score_program(_PROFITABLE, always_pass)
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
        assert r.mean_metric == pytest.approx(
            score_program(_PROFITABLE, always_pass).mean_metric
        )

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


def _row(ticker: str, market: str = "us") -> dict:
    return {"track": "claude", "ticker": ticker, "market": market,
            "price": 100.0, "atr": 3.0, "timestamp": None, "entry_inputs": {}}


class TestRowSelection:
    """One blob per (track, ticker) per day means a name decided daily for six
    weeks contributes ~45 correlated rows. Letting it dominate would inflate n
    without adding independent evidence."""

    def test_most_decided_tickers_come_first(self):
        rows = [_row("RARE")] + [_row("COMMON")] * 20 + [_row("MID")] * 5
        picked = select_decision_rows(rows, limit=100, max_per_ticker=99, max_tickers=2)
        tickers = {r["ticker"] for r in picked}
        assert tickers == {"COMMON", "MID"}
        assert "RARE" not in tickers

    def test_no_single_ticker_dominates(self):
        rows = [_row("COMMON")] * 50 + [_row("OTHER")] * 50
        picked = select_decision_rows(rows, limit=100, max_per_ticker=5, max_tickers=10)
        counts = {}
        for r in picked:
            counts[r["ticker"]] = counts.get(r["ticker"], 0) + 1
        assert all(c <= 5 for c in counts.values())

    def test_distinct_tickers_are_capped(self):
        rows = [_row(f"T{i}") for i in range(500)]
        picked = select_decision_rows(rows, limit=500, max_per_ticker=5, max_tickers=20)
        assert len({r["ticker"] for r in picked}) <= 20

    def test_limit_is_respected(self):
        rows = [_row(f"T{i}") for i in range(500)]
        assert len(select_decision_rows(rows, limit=30, max_per_ticker=5, max_tickers=150)) == 30

    def test_empty_input_is_empty_output(self):
        assert select_decision_rows([], limit=10, max_per_ticker=5, max_tickers=10) == []


class TestBatchFetching:
    """One chunked download per market, not one call per ticker — the
    per-ticker path routes Nordic through Alpha Vantage's 25/day free tier."""

    def test_one_call_per_market_regardless_of_ticker_count(self, monkeypatch):
        calls = {"nordic": 0, "eu": 0, "us": 0}

        def _make(market):
            def _fetch(tickers):
                calls[market] += 1
                return {t: "df" for t in tickers}
            return _fetch

        import src.data.market_data as md
        monkeypatch.setattr(md, "fetch_batch_nordic", _make("nordic"))
        monkeypatch.setattr(md, "fetch_batch_eu", _make("eu"))
        monkeypatch.setattr(md, "fetch_batch_us", _make("us"))

        rows = ([_row(f"N{i}", "nordic") for i in range(80)]
                + [_row(f"U{i}", "us") for i in range(120)])
        prices = _batch_prices(rows)

        assert calls == {"nordic": 1, "eu": 0, "us": 1}
        assert len(prices) == 200

    def test_a_failing_market_does_not_abort_the_others(self, monkeypatch):
        import src.data.market_data as md

        def _boom(_):
            raise RuntimeError("provider down")

        monkeypatch.setattr(md, "fetch_batch_nordic", _boom)
        monkeypatch.setattr(md, "fetch_batch_us", lambda t: {x: "df" for x in t})

        prices = _batch_prices([_row("N1", "nordic"), _row("U1", "us")])
        assert "U1" in prices and "N1" not in prices

    def test_unknown_market_is_skipped_not_fatal(self, monkeypatch):
        import src.data.market_data as md
        monkeypatch.setattr(md, "fetch_batch_us", lambda t: {x: "df" for x in t})
        prices = _batch_prices([_row("X1", "crypto"), _row("U1", "us")])
        assert "U1" in prices and "X1" not in prices


class TestHeadroom:
    """PASS no longer scores a flat 0.5, so there is no fixed baseline to read a
    result against. The oracle gap is the replacement, and it is the number that
    says whether a corpus can distinguish prompts at all."""

    def test_positive_when_the_corpus_has_edge(self):
        results = [
            score_program(_PROFITABLE, oracle, "oracle"),
            score_program(_PROFITABLE, always_buy, "always-buy"),
            score_program(_PROFITABLE, always_pass, "always-pass"),
        ]
        assert headroom(results) > 0

    def test_near_zero_when_every_setup_is_identical(self):
        # Nothing to discriminate: every example carries the same outcome, so
        # perfect foresight is worth nothing over the right blanket rule.
        flat = [_ex("PASS", -1.0) for _ in range(20)]
        results = [
            score_program(flat, oracle, "oracle"),
            score_program(flat, always_buy, "always-buy"),
            score_program(flat, always_pass, "always-pass"),
        ]
        assert headroom(results) == pytest.approx(0.0, abs=1e-9)

    def test_none_without_an_oracle_to_compare_against(self):
        assert headroom([score_program(_PROFITABLE, always_buy, "always-buy")]) is None
