from datetime import date

import pytest

from src.agent.session_coverage import freeze_sessions, validate_sessions


def test_us_holiday_and_half_day_are_distinguished():
    coverage = freeze_sessions("AAPL", "us", date(2026, 12, 23), date(2026, 12, 28))
    assert coverage["expected_sessions"] == ["2026-12-24", "2026-12-28"]
    validate_sessions(coverage, coverage["expected_sessions"])


def test_listing_exchange_controls_holidays():
    coverage = freeze_sessions("VOLV-B.ST", "nordic", date(2026, 12, 23), date(2026, 12, 28))
    assert coverage["calendar"] == "XSTO"
    assert coverage["expected_sessions"] == ["2026-12-28"]


def test_missing_middle_session_is_rejected_even_with_correct_boundaries():
    coverage = freeze_sessions("AAPL", "us", date(2026, 9, 7), date(2026, 9, 11))
    with pytest.raises(ValueError, match="missing=.*2026-09-09"):
        validate_sessions(coverage, ["2026-09-08", "2026-09-10", "2026-09-11"])


def test_duplicate_and_non_session_bars_are_rejected():
    coverage = freeze_sessions("AAPL", "us", date(2026, 12, 23), date(2026, 12, 28))
    expected = coverage["expected_sessions"]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_sessions(coverage, expected + expected[:1])
    with pytest.raises(ValueError, match="unexpected=.*2026-12-25"):
        validate_sessions(coverage, expected + ["2026-12-25"])


def test_unknown_listing_is_not_assigned_a_generic_regional_calendar():
    with pytest.raises(ValueError, match="No verified exchange calendar"):
        freeze_sessions("UNKNOWN.ZZ", "eu", date(2026, 9, 1), date(2026, 9, 15))


def test_frozen_dates_validate_without_reconsulting_calendar(monkeypatch):
    coverage = freeze_sessions("AAPL", "us", date(2026, 9, 7), date(2026, 9, 11))
    monkeypatch.setattr("src.agent.session_coverage.xcals.get_calendar", lambda *a, **kw: pytest.fail("calendar changed"))
    validate_sessions(coverage, coverage["expected_sessions"])
