from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

import src.agent.memory as _memory_module
from config.settings import settings


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Point the DB at a temp file and initialize the schema."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(type(settings), "db_path", property(lambda self: db_file))
    from src.db import init_db
    init_db()
    return db_file


@pytest.fixture()
def tmp_heuristics(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "heuristics_dir", property(lambda self: tmp_path))
    _memory_module._stores.clear()
    yield tmp_path
    _memory_module._stores.clear()


def _seed_decision(ticker: str, price: float, days_ago: int, entry_inputs=None,
                   track: str = "claude", action: str = "PASS", atr=None):
    from src.db import Decision, get_session
    session = get_session()
    try:
        session.add(Decision(
            market="us", track=track, ticker=ticker, action=action,
            confidence=0.5, regime="trending", reasoning="test",
            price=price, atr=atr, entry_inputs=entry_inputs,
            timestamp=datetime.utcnow() - timedelta(days=days_ago),
        ))
        session.commit()
    finally:
        session.close()


def _price_df(start: datetime, days: int, close: float) -> pd.DataFrame:
    idx = pd.date_range(start=start.date(), periods=days, freq="D")
    return pd.DataFrame({"Close": [close] * days}, index=idx)


def _ohlc_df(start: datetime, bars: list[tuple[float, float, float]]) -> pd.DataFrame:
    """DataFrame from (high, low, close) tuples, one bar per day."""
    idx = pd.date_range(start=start.date(), periods=len(bars), freq="D")
    return pd.DataFrame(
        {
            "High": [b[0] for b in bars],
            "Low": [b[1] for b in bars],
            "Close": [b[2] for b in bars],
        },
        index=idx,
    )


_INPUTS = {"technicals": "RSI=55", "regime": "trending", "news_summary": "n",
           "macro_context": "m", "heuristics": "h"}


class TestCounterfactualExamples:
    def _build(self, max_examples: int = 30):
        from src.scheduler.optimizer import _build_counterfactual_examples
        with patch("src.scheduler.optimizer._make_example",
                   side_effect=lambda inputs, action, pnl: {"inputs": inputs, "action": action, "pnl": pnl}):
            return _build_counterfactual_examples("claude", max_examples)

    def test_missed_winner_labeled_buy(self, tmp_db):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS)
        df = _price_df(datetime.utcnow() - timedelta(days=29), 40, close=110.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            examples = self._build()
        assert len(examples) == 1
        assert examples[0]["action"] == "BUY"
        assert examples[0]["pnl"] == pytest.approx(0.10)
        assert examples[0]["inputs"] == _INPUTS

    def test_correct_pass_labeled_pass(self, tmp_db):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS)
        df = _price_df(datetime.utcnow() - timedelta(days=29), 40, close=94.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            examples = self._build()
        assert len(examples) == 1
        assert examples[0]["action"] == "PASS"
        assert examples[0]["pnl"] == pytest.approx(-0.06)

    def test_ambiguous_drift_skipped(self, tmp_db):
        # +1% forward return: between 0 and the 3% threshold → no clean label
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS)
        df = _price_df(datetime.utcnow() - timedelta(days=29), 40, close=101.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            assert self._build() == []

    def test_recent_decisions_excluded(self, tmp_db):
        # Younger than the horizon — no forward window yet
        _seed_decision("AAPL", price=100.0, days_ago=2, entry_inputs=_INPUTS)
        df = _price_df(datetime.utcnow() - timedelta(days=10), 20, close=120.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            assert self._build() == []

    def test_decisions_without_inputs_excluded(self, tmp_db):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=None)
        df = _price_df(datetime.utcnow() - timedelta(days=29), 40, close=120.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            assert self._build() == []

    def test_other_track_excluded(self, tmp_db):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS, track="gpt")
        df = _price_df(datetime.utcnow() - timedelta(days=29), 40, close=120.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            assert self._build() == []  # building for "claude"

    def test_cap_respected(self, tmp_db):
        for i in range(5):
            _seed_decision(f"TICK{i}", price=100.0, days_ago=30 + i, entry_inputs=_INPUTS)
        df = _price_df(datetime.utcnow() - timedelta(days=40), 50, close=110.0)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            examples = self._build(max_examples=2)
        assert len(examples) == 2

    def test_missing_price_data_skipped(self, tmp_db):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS)
        with patch("src.data.market_data.fetch_ohlcv", return_value=None):
            assert self._build() == []


class TestCounterfactualPathSimulation:
    """With ATR persisted, the label comes from the simulated stop/target path,
    not the horizon close — a rally that traded through its stop first is a
    correct PASS, not a missed BUY. ATR=2 → stop 97, target 106 (RRR 2.0)."""

    def _build(self):
        from src.scheduler.optimizer import _build_counterfactual_examples
        with patch("src.scheduler.optimizer._make_example",
                   side_effect=lambda inputs, action, pnl: {"action": action, "pnl": pnl}):
            return _build_counterfactual_examples("claude", 30)

    def _run(self, bars: list[tuple[float, float, float]]):
        _seed_decision("AAPL", price=100.0, days_ago=30, entry_inputs=_INPUTS, atr=2.0)
        df = _ohlc_df(datetime.utcnow() - timedelta(days=29), bars)
        with patch("src.data.market_data.fetch_ohlcv", return_value=df):
            return self._build()

    def test_target_hit_labels_missed_buy(self, tmp_db):
        examples = self._run([(101, 99, 100), (103, 100, 102), (107, 102, 106), (110, 106, 109)])
        assert len(examples) == 1
        assert examples[0]["action"] == "BUY"
        assert examples[0]["pnl"] == pytest.approx(0.06)  # target return, not final close

    def test_stop_first_rally_is_correct_pass(self, tmp_db):
        # Dips through the stop (97) before rallying to 115 — horizon-close
        # labeling would call this a missed +15% winner; the path says stop-out.
        examples = self._run([(100, 96, 98), (105, 98, 104), (116, 104, 115), (116, 114, 115)])
        assert len(examples) == 1
        assert examples[0]["action"] == "PASS"
        assert examples[0]["pnl"] == pytest.approx(-0.03)  # stop return

    def test_both_hit_same_bar_stop_wins(self, tmp_db):
        examples = self._run([(107, 96, 105), (108, 104, 107), (108, 105, 107)])
        assert len(examples) == 1
        assert examples[0]["action"] == "PASS"

    def test_no_exit_hit_falls_back_to_close_label(self, tmp_db):
        # Never touches 97 or 106; drifts up 4% → BUY via the threshold fallback
        examples = self._run([(102, 99, 101), (103, 100, 102), (105, 102, 104), (105, 103, 104)])
        assert len(examples) == 1
        assert examples[0]["action"] == "BUY"
        assert examples[0]["pnl"] == pytest.approx(0.04)


class TestDecisionPersistenceDedupe:
    def test_one_inputs_blob_per_track_ticker_day(self, tmp_db):
        from src.db import Decision, get_session
        from src.scheduler.scan_loop import _persist_decisions

        d = {"track": "claude", "ticker": "AAPL", "action": "PASS",
             "confidence": 0.4, "regime": "trending", "reasoning": "r",
             "price": 100.0, "entry_inputs": _INPUTS}
        _persist_decisions("us", [dict(d)])
        _persist_decisions("us", [dict(d)])  # next 15-min scan, same day

        session = get_session()
        try:
            rows = session.query(Decision).filter(Decision.ticker == "AAPL").all()
            assert len(rows) == 2
            with_inputs = [r for r in rows if r.entry_inputs is not None]
            assert len(with_inputs) == 1
            assert with_inputs[0].price == pytest.approx(100.0)
        finally:
            session.close()

    def test_different_tracks_each_keep_inputs(self, tmp_db):
        from src.db import Decision, get_session
        from src.scheduler.scan_loop import _persist_decisions

        decisions = [
            {"track": t, "ticker": "AAPL", "action": "PASS", "price": 100.0,
             "entry_inputs": _INPUTS}
            for t in ("claude", "gpt")
        ]
        _persist_decisions("us", decisions)

        session = get_session()
        try:
            with_inputs = (
                session.query(Decision)
                .filter(Decision.entry_inputs.isnot(None))
                .all()
            )
            assert {r.track for r in with_inputs} == {"claude", "gpt"}
        finally:
            session.close()


class TestDecisionsMigration:
    def test_old_schema_gains_new_columns(self, tmp_path, monkeypatch):
        db_file = tmp_path / "old.db"
        monkeypatch.setattr(type(settings), "db_path", property(lambda self: db_file))

        import sqlite3
        conn = sqlite3.connect(db_file)
        conn.execute("""CREATE TABLE decisions (
            id INTEGER PRIMARY KEY, timestamp DATETIME, market VARCHAR(10) NOT NULL,
            track VARCHAR(10) NOT NULL, ticker VARCHAR(20) NOT NULL,
            action VARCHAR(10) NOT NULL, confidence FLOAT, rrr FLOAT,
            regime VARCHAR(20), reasoning TEXT, block_reason TEXT)""")
        conn.commit()
        conn.close()

        from src.db import init_db
        init_db()

        conn = sqlite3.connect(db_file)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}
        conn.close()
        assert "price" in cols
        assert "entry_inputs" in cols


class TestHeuristicOutcomeFeedback:
    def test_win_raises_quality(self, tmp_heuristics):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=5.0)

        updated = store.record_outcome([hid], pnl_pct=0.05)

        assert updated == 1
        data = json.loads((tmp_heuristics / "claude" / f"{hid}.json").read_text())
        assert data["quality_score"] == pytest.approx(5.5)
        assert data["outcome_count"] == 1
        assert data["cumulative_pnl_pct"] == pytest.approx(0.05)

    def test_loss_lowers_quality(self, tmp_heuristics):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=5.0)

        store.record_outcome([hid], pnl_pct=-0.08)

        data = json.loads((tmp_heuristics / "claude" / f"{hid}.json").read_text())
        assert data["quality_score"] == pytest.approx(4.2)

    def test_delta_clamped_to_one(self, tmp_heuristics):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=5.0)

        store.record_outcome([hid], pnl_pct=0.50)  # would be +5 unclamped

        data = json.loads((tmp_heuristics / "claude" / f"{hid}.json").read_text())
        assert data["quality_score"] == pytest.approx(6.0)

    def test_quality_bounded_zero_to_ten(self, tmp_heuristics):
        from src.agent.memory import get_store
        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=0.3)

        store.record_outcome([hid], pnl_pct=-0.20)

        data = json.loads((tmp_heuristics / "claude" / f"{hid}.json").read_text())
        assert data["quality_score"] == 0.0

    def test_missing_heuristic_skipped(self, tmp_heuristics):
        from src.agent.memory import get_store
        store = get_store("claude")
        assert store.record_outcome(["nonexistent-id"], pnl_pct=0.05) == 0

    def test_close_hook_records_outcome(self, tmp_heuristics):
        from types import SimpleNamespace
        from src.agent.memory import get_store
        from src.scheduler.scan_loop import _record_heuristic_outcome

        store = get_store("claude")
        hid = store.save(trigger="A", action="B", quality_score=5.0)
        closed = SimpleNamespace(
            trade_id=1, pnl_pct=0.04,
            entry_inputs={**_INPUTS, "heuristic_ids": [hid]},
        )

        _record_heuristic_outcome("claude", closed)

        data = json.loads((tmp_heuristics / "claude" / f"{hid}.json").read_text())
        assert data["quality_score"] == pytest.approx(5.4)


class TestNewsPrefilterCompanyName:
    def test_nordic_headline_matches_company_name(self):
        from src.agent.news_analyzer import _prefilter
        articles = [{"headline": "Volvo lanserar ny elektrisk lastbilsplattform"}]
        assert _prefilter("VOLV-B.ST", articles) == articles

    def test_share_class_suffix_stripped(self):
        from src.agent.news_analyzer import _company_name_term
        # universe.csv name is "Ericsson B" — headlines just say Ericsson
        assert _company_name_term("ERIC-B.ST") == "ericsson"

    def test_unrelated_headline_still_dropped(self):
        from src.agent.news_analyzer import _prefilter
        articles = [{"headline": "Ny rekordnotering på Stockholmsbörsen"}]
        assert _prefilter("VOLV-B.ST", articles) == []

    def test_unknown_ticker_falls_back_to_keywords(self):
        from src.agent.news_analyzer import _prefilter
        articles = [{"headline": "Company earnings beat expectations"}]
        assert _prefilter("ZZZZ", articles) == articles
