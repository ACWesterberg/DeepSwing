"""Freeze expected exchange sessions and reject incomplete daily price paths."""
from datetime import date, timedelta

import exchange_calendars as xcals


CALENDARS = {
    "ST": "XSTO", "STO": "XSTO", "OL": "XOSL", "HE": "XHEL", "CO": "XCSE",
    "L": "XLON", "DE": "XETR", "PA": "XPAR", "AS": "XAMS", "BR": "XBRU",
    "LS": "XLIS", "MC": "XMAD", "SW": "XSWX", "WA": "XWAR", "VI": "XWBO",
    "MI": "XMIL", "IR": "XDUB",
}


def freeze_sessions(ticker: str, market: str, start: date, end: date) -> dict:
    suffix = ticker.upper().rsplit(".", 1)[-1] if "." in ticker else None
    calendar = CALENDARS.get(suffix)
    if calendar is None and market == "us" and suffix is None:
        calendar = "XNYS"  # US primary equity listings share this session calendar.
    if calendar is None:
        raise ValueError(f"No verified exchange calendar for {ticker} ({market})")
    if end <= start:
        raise ValueError("Evaluation horizon must follow the decision date")
    first = start + timedelta(days=1)
    cal = xcals.get_calendar(calendar, start=start.isoformat(), end=(end + timedelta(days=7)).isoformat())
    sessions = [stamp.date().isoformat() for stamp in cal.sessions_in_range(first.isoformat(), end.isoformat())]
    if not sessions:
        raise ValueError("Evaluation horizon contains no exchange sessions")
    return {"calendar": calendar, "calendar_version": xcals.__version__,
            "start_exclusive": start.isoformat(), "end_inclusive": end.isoformat(),
            "expected_sessions": sessions}


def validate_sessions(coverage: dict, dates) -> None:
    observed = [str(value)[:10] for value in dates]
    expected = coverage["expected_sessions"]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate daily session bars")
    missing, unexpected = set(expected) - set(observed), set(observed) - set(expected)
    if missing or unexpected:
        raise ValueError(f"Incomplete exchange sessions: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
