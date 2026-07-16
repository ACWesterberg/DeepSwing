from __future__ import annotations

from datetime import datetime

import pytest

from config.settings import settings


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(type(settings), "db_path", property(lambda self: db_file))
    from src.db import init_db
    init_db()
    return db_file


@pytest.fixture()
def wm(monkeypatch):
    """watch_monitor module with all network/LLM edges stubbed to quiet defaults."""
    import src.scheduler.watch_monitor as wm

    monkeypatch.setattr(wm, "get_current_price", lambda t, m: None)
    monkeypatch.setattr(wm, "_previous_close", lambda t, m: None)
    monkeypatch.setattr(wm, "fetch_news_for_ticker", lambda t, m: [])
    monkeypatch.setattr(wm, "get_insider_summary", lambda t, m: "")
    monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("neutral", ""))
    monkeypatch.setattr(wm, "is_market_open", lambda m: True)
    monkeypatch.setattr(wm, "send_telegram", lambda text: True)
    return wm


def _add_watch(ticker="AAPL", market="us", baselined=True):
    from src.db import WatchedTicker, get_session
    session = get_session()
    try:
        session.add(WatchedTicker(ticker=ticker, market=market, baselined=baselined))
        session.commit()
    finally:
        session.close()


def _alerts():
    from src.db import WatchAlert, get_session
    session = get_session()
    try:
        return [a.to_dict() for a in session.query(WatchAlert).order_by(WatchAlert.id).all()]
    finally:
        session.close()


def _watch(ticker="AAPL"):
    from src.db import WatchedTicker, get_session
    session = get_session()
    try:
        row = session.get(WatchedTicker, ticker)
        session.refresh(row)
        return row
    finally:
        session.close()


class TestMoveAlertDue:
    def test_below_threshold_is_silent(self, wm):
        assert not wm._move_alert_due(0.02, "2026-07-16", None, None)

    def test_first_alert_at_threshold(self, wm):
        assert wm._move_alert_due(0.031, "2026-07-16", None, None)
        assert wm._move_alert_due(-0.031, "2026-07-16", None, None)

    def test_no_realert_without_extension(self, wm):
        assert not wm._move_alert_due(0.035, "2026-07-16", "2026-07-16", 0.032)

    def test_realert_after_step_extension(self, wm):
        assert wm._move_alert_due(0.055, "2026-07-16", "2026-07-16", 0.032)

    def test_realert_on_direction_flip(self, wm):
        assert wm._move_alert_due(-0.04, "2026-07-16", "2026-07-16", 0.04)

    def test_new_day_resets(self, wm):
        assert wm._move_alert_due(0.031, "2026-07-17", "2026-07-16", 0.08)


class TestBaseline:
    def test_first_pass_records_state_without_alerting(self, tmp_db, wm, monkeypatch):
        _add_watch(baselined=False)
        monkeypatch.setattr(wm, "get_current_price", lambda t, m: 105.0)
        monkeypatch.setattr(wm, "_previous_close", lambda t, m: 100.0)
        monkeypatch.setattr(
            wm, "fetch_news_for_ticker",
            lambda t, m: [{"headline": "Old news", "source_url": "u1", "published_at": None}],
        )
        monkeypatch.setattr(wm, "get_insider_summary", lambda t, m: "CEO bought 10k shares")
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("bullish", "looks great"))

        result = wm.run_watch_monitor()
        assert result["alerts"] == []

        row = _watch()
        assert row.baselined is True
        assert len(row.seen_news_hashes) == 1
        assert row.insider_hash is not None
        assert row.last_price == 105.0
        assert row.last_move_pct == pytest.approx(0.05)


class TestNewsAlerts:
    def test_new_directional_headline_pings_once(self, tmp_db, wm, monkeypatch):
        _add_watch()
        articles = [{"headline": "Massive earnings beat", "source_url": "u2", "published_at": None}]
        monkeypatch.setattr(wm, "fetch_news_for_ticker", lambda t, m: articles)
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("bullish", "earnings beat"))
        sent = []
        monkeypatch.setattr(wm, "send_telegram", lambda text: sent.append(text) or True)

        first = wm.run_watch_monitor()
        assert len(first["alerts"]) == 1
        assert first["alerts"][0]["verdict"] == "bullish"
        assert "Massive earnings beat" in sent[0]

        second = wm.run_watch_monitor()
        assert second["alerts"] == []
        assert len(_alerts()) == 1

    def test_neutral_news_is_marked_seen_but_silent(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(
            wm, "fetch_news_for_ticker",
            lambda t, m: [{"headline": "Company attends conference", "source_url": "u3", "published_at": None}],
        )
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("neutral", ""))

        result = wm.run_watch_monitor()
        assert result["alerts"] == []
        assert len(_watch().seen_news_hashes) == 1

    def test_stale_articles_are_ignored(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(
            wm, "fetch_news_for_ticker",
            lambda t, m: [{"headline": "Ancient news", "source_url": "u4", "published_at": "2026-01-01T00:00:00Z"}],
        )
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("bullish", "old"))

        result = wm.run_watch_monitor()
        assert result["alerts"] == []
        assert _watch().seen_news_hashes in (None, [])


class TestMoveAlerts:
    def test_large_move_pings_and_dedupes_same_day(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(wm, "get_current_price", lambda t, m: 105.0)
        monkeypatch.setattr(wm, "_previous_close", lambda t, m: 100.0)

        first = wm.run_watch_monitor()
        assert len(first["alerts"]) == 1
        assert first["alerts"][0]["kind"] == "move"
        assert first["alerts"][0]["verdict"] == "bullish"

        second = wm.run_watch_monitor()
        assert second["alerts"] == []

        # extends past the re-alert step → pings again
        monkeypatch.setattr(wm, "get_current_price", lambda t, m: 108.0)
        third = wm.run_watch_monitor()
        assert len(third["alerts"]) == 1

    def test_no_move_alert_when_market_closed(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(wm, "get_current_price", lambda t, m: 105.0)
        monkeypatch.setattr(wm, "_previous_close", lambda t, m: 100.0)
        monkeypatch.setattr(wm, "is_market_open", lambda m: False)

        result = wm.run_watch_monitor()
        assert result["alerts"] == []
        # snapshot still refreshed for the dashboard
        assert _watch().last_price == 105.0


class TestInsiderAlerts:
    def test_insider_change_pings(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(wm, "get_insider_summary", lambda t, m: "CFO sold 50k shares")
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("bearish", "big sale"))

        first = wm.run_watch_monitor()
        assert len(first["alerts"]) == 1
        assert first["alerts"][0]["kind"] == "insider"

        second = wm.run_watch_monitor()
        assert second["alerts"] == []

    def test_buys_only_skips_bearish(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(settings, "watch_insider_buys_only", True)
        monkeypatch.setattr(wm, "get_insider_summary", lambda t, m: "CFO sold 50k shares")
        monkeypatch.setattr(wm, "classify_watch_event", lambda t, k, c: ("bearish", "big sale"))

        result = wm.run_watch_monitor()
        assert result["alerts"] == []
        # hash still updated so the same event never re-pings if the knob flips
        assert _watch().insider_hash is not None


class TestDelivery:
    def test_failed_send_is_recorded_undelivered(self, tmp_db, wm, monkeypatch):
        _add_watch()
        monkeypatch.setattr(wm, "get_current_price", lambda t, m: 110.0)
        monkeypatch.setattr(wm, "_previous_close", lambda t, m: 100.0)
        monkeypatch.setattr(wm, "send_telegram", lambda text: False)

        result = wm.run_watch_monitor()
        assert result["alerts"][0]["delivered"] is False
        assert _alerts()[0]["delivered"] is False

    def test_telegram_noop_without_keys(self, monkeypatch):
        from src.notify import telegram
        monkeypatch.setattr(settings, "telegram_bot_token", "")
        monkeypatch.setattr(settings, "telegram_chat_id", "")
        assert telegram.telegram_configured() is False
        assert telegram.send_telegram("hello") is False


class TestPrune:
    def test_alert_log_capped(self, tmp_db, wm, monkeypatch):
        from src.db import WatchAlert, get_session
        monkeypatch.setattr(settings, "watch_alerts_retention", 10)
        session = get_session()
        try:
            for i in range(25):
                session.add(WatchAlert(ticker="AAPL", kind="news", verdict="bullish", message=str(i)))
            session.commit()
            wm._prune_alerts(session)
        finally:
            session.close()
        remaining = _alerts()
        assert len(remaining) == 10
        assert remaining[0]["message"] == "15"
