from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import src.agent.memory as _memory_module
from config.settings import settings


@pytest.fixture(autouse=True)
def _store_dir(tmp_path, monkeypatch):
    _memory_module._stores.clear()
    monkeypatch.setattr(type(settings), "heuristics_dir", property(lambda self: tmp_path))
    yield
    _memory_module._stores.clear()


def _read(tmp_path, track, hid):
    return json.loads((tmp_path / track / f"{hid}.json").read_text())


def _write(tmp_path, track, hid, **fields):
    path = tmp_path / track / f"{hid}.json"
    data = json.loads(path.read_text())
    data.update(fields)
    path.write_text(json.dumps(data))


class TestCorePromotion:
    """Access count measures how often a rule was shown, not whether it helped."""

    def test_popular_but_poor_is_not_promoted(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=3.0)
        _write(tmp_path, "claude", hid, access_count=50)

        promoted, _ = store.promote_core()
        assert promoted == 0
        assert _read(tmp_path, "claude", hid)["is_core"] is False

    def test_popular_and_proven_is_promoted(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=7.0)
        _write(tmp_path, "claude", hid, access_count=10)

        promoted, _ = store.promote_core()
        assert promoted == 1
        assert _read(tmp_path, "claude", hid)["is_core"] is True

    def test_core_rule_that_decays_is_demoted(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=7.0)
        _write(tmp_path, "claude", hid, access_count=20, is_core=True, quality_score=3.0)

        promoted, demoted = store.promote_core()
        assert (promoted, demoted) == (0, 1)
        assert _read(tmp_path, "claude", hid)["is_core"] is False

    def test_core_rule_still_working_keeps_the_flag(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=7.0)
        _write(tmp_path, "claude", hid, access_count=20, is_core=True)

        assert store.promote_core() == (0, 0)
        assert _read(tmp_path, "claude", hid)["is_core"] is True


class TestCorroborationWeighting:
    """A heuristic is extracted from one trade, so a new one is an untested
    guess and should rank under rules that have actually been measured."""

    def test_tested_rule_outranks_untested_of_equal_quality(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        untested = store.save(trigger="fresh idea", action="do X",
                              quality_score=7.0, regime="trending", market="us")
        tested = store.save(trigger="proven rule", action="do Y",
                            quality_score=7.0, regime="trending", market="us")
        _write(tmp_path, "claude", tested,
               outcome_count=_memory_module.MIN_CORROBORATION)

        top = store.retrieve(ticker="AAPL", regime="trending", market="us", top_k=2)
        assert [h["id"] for h in top][0] == tested
        assert untested in [h["id"] for h in top]  # still retrievable, just lower

    def test_untested_still_used_when_it_is_all_there_is(self, tmp_path):
        from src.agent.memory import get_store
        store = get_store("claude")
        store.save(trigger="only rule", action="do X", regime="trending", market="us")
        assert len(store.retrieve(ticker="AAPL", regime="trending", market="us")) == 1

    def test_strong_untested_can_still_beat_weak_tested(self, tmp_path):
        # The penalty ranks unproven rules down; it must not shut them out, or
        # a genuinely good new rule could never accumulate the outcomes that
        # would prove it.
        from src.agent.memory import get_store
        store = get_store("claude")
        weak = store.save(trigger="weak", action="Y", quality_score=2.0,
                          regime="trending", market="us")
        strong = store.save(trigger="strong", action="X", quality_score=9.0,
                            regime="trending", market="us")
        _write(tmp_path, "claude", weak, outcome_count=5)

        top = store.retrieve(ticker="AAPL", regime="trending", market="us", top_k=2)
        assert top[0]["id"] == strong


class TestSkippedSetupScoring:
    """Heuristics were only ever scored on trades that opened, so a rule that
    argued for passing was never held to account either way."""

    @pytest.fixture
    def tmp_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(type(settings), "db_path",
                            property(lambda self: tmp_path / "test.db"))
        from src.db import init_db
        init_db()
        yield tmp_path

    def _seed(self, ticker, hids, action="PASS", price=100.0, atr=2.0, days_ago=30):
        from src.db import Decision, get_session
        session = get_session()
        try:
            session.add(Decision(
                market="us", track="claude", ticker=ticker, action=action,
                confidence=0.5, regime="trending", reasoning="t",
                price=price, atr=atr,
                entry_inputs={"technicals": "x", "heuristic_ids": hids},
                timestamp=datetime.utcnow() - timedelta(days=days_ago),
            ))
            session.commit()
        finally:
            session.close()

    def _bars(self, closes):
        idx = pd.date_range(start=(datetime.utcnow() - timedelta(days=29)).date(),
                            periods=len(closes), freq="D")
        return pd.DataFrame({"Close": closes}, index=idx)

    def _run(self):
        from src.scheduler.optimizer import score_heuristics_from_decisions
        return score_heuristics_from_decisions("claude")

    def test_passing_on_a_winner_costs_the_heuristic(self, tmp_db):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="skip these", quality_score=5.0)
        self._seed("AAPL", [hid])

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([110.0] * 20)):
            assert self._run() == 1

        assert _read(tmp_db, "claude", hid)["quality_score"] < 5.0

    def test_passing_on_a_loser_rewards_the_heuristic(self, tmp_db):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="skip these", quality_score=5.0)
        self._seed("AAPL", [hid])

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([92.0] * 20)):
            assert self._run() == 1

        assert _read(tmp_db, "claude", hid)["quality_score"] > 5.0

    def test_blocked_buy_scores_in_the_buy_direction(self, tmp_db):
        # BLOCKED carried the model's intent to buy, so a winner vindicates the
        # heuristics behind it — the opposite sign to a PASS.
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="take these", quality_score=5.0)
        self._seed("AAPL", [hid], action="BLOCKED")

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([110.0] * 20)):
            assert self._run() == 1

        assert _read(tmp_db, "claude", hid)["quality_score"] > 5.0

    def test_scoring_is_idempotent(self, tmp_db):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="skip", quality_score=5.0)
        self._seed("AAPL", [hid])

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([110.0] * 20)):
            assert self._run() == 1
            after_first = _read(tmp_db, "claude", hid)["quality_score"]
            # record_outcome has no idempotency of its own; the DB flag is what
            # stops a weekly re-run compounding the same evidence.
            assert self._run() == 0

        assert _read(tmp_db, "claude", hid)["quality_score"] == after_first

    def test_outcome_count_accrues_from_skipped_setups(self, tmp_db):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="skip", quality_score=5.0)
        self._seed("AAPL", [hid])

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([92.0] * 20)):
            self._run()

        # This is what lets a rule reach MIN_CORROBORATION without having to
        # have opened a position first.
        assert _read(tmp_db, "claude", hid)["outcome_count"] == 1

    def test_recent_decisions_are_left_alone(self, tmp_db):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="skip", quality_score=5.0)
        self._seed("AAPL", [hid], days_ago=2)   # inside the horizon

        with patch("src.data.market_data.fetch_ohlcv", return_value=self._bars([110.0] * 20)):
            assert self._run() == 0
        assert _read(tmp_db, "claude", hid)["quality_score"] == 5.0

    def test_legacy_rows_without_ids_are_marked_not_refetched(self, tmp_db):
        from src.db import Decision, get_session
        session = get_session()
        try:
            session.add(Decision(
                market="us", track="claude", ticker="AAPL", action="PASS",
                price=100.0, atr=2.0, entry_inputs={"technicals": "x"},
                timestamp=datetime.utcnow() - timedelta(days=30),
            ))
            session.commit()
        finally:
            session.close()

        assert self._run() == 0
        session = get_session()
        try:
            assert session.query(Decision).first().heuristics_scored is True
        finally:
            session.close()
