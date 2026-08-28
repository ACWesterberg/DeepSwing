from __future__ import annotations

from datetime import date, datetime

import pytest

from src.analysis.event_model import EventContract
from src.data import weather_forecast as wf

NOW = datetime(2026, 8, 27, 12, 0, 0)

POINTS_PAYLOAD = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/OKX/33,37/forecast",
        "gridId": "OKX",
    }
}

FORECAST_PAYLOAD = {
    "properties": {
        "periods": [
            {"startTime": "2026-08-27T06:00:00-04:00", "isDaytime": True,
             "temperature": 84, "temperatureUnit": "F"},
            {"startTime": "2026-08-27T18:00:00-04:00", "isDaytime": False,
             "temperature": 68, "temperatureUnit": "F"},
            {"startTime": "2026-08-28T06:00:00-04:00", "isDaytime": True,
             "temperature": 81, "temperatureUnit": "F"},
            {"startTime": "2026-08-28T18:00:00-04:00", "isDaytime": False,
             "temperature": 65, "temperatureUnit": "F"},
        ]
    }
}


@pytest.fixture(autouse=True)
def _clean_state():
    wf.reset_breaker()
    yield
    wf.reset_breaker()


def contract(
    ticker: str = "KXHIGHNY-26AUG28-B82",
    event_ticker: str = "KXHIGHNY-26AUG28",
    series_ticker: str = "KXHIGHNY",
    close_time: datetime | None = None,
) -> EventContract:
    return EventContract(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title="NYC high",
        yes_bid=0.20, yes_ask=0.24, last_price=0.22,
        volume=100, open_interest=1000,
        close_time=close_time or datetime(2026, 8, 28, 23, 59),
        strike_type="between", floor_strike=82, cap_strike=83,
    )


def install_json(monkeypatch, by_url: dict, calls: list | None = None):
    def fake_get_json(url, params=None):
        if calls is not None:
            calls.append(url)
        for fragment, payload in by_url.items():
            if fragment in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(wf, "_get_json", fake_get_json)


class TestTargetDate:
    def test_local_evening_close_maps_to_that_day(self):
        # 23:59 UTC on the 28th is 19:59 local in New York — still the 28th.
        assert wf.target_date(contract()) == date(2026, 8, 28)

    def test_close_at_local_midnight_does_not_roll_forward(self):
        # 04:00 UTC on the 29th is exactly midnight local ending the 28th.
        c = contract(close_time=datetime(2026, 8, 29, 4, 0))
        assert wf.target_date(c) == date(2026, 8, 28)

    def test_timezone_is_per_station(self):
        # 03:00 UTC is 20:00 the previous day in Los Angeles, 23:00 in New York.
        close = datetime(2026, 8, 29, 3, 0)
        ny = contract(series_ticker="KXHIGHNY", close_time=close)
        la = contract(series_ticker="KXHIGHLAX", close_time=close)
        assert wf.target_date(ny) == date(2026, 8, 28)
        assert wf.target_date(la) == date(2026, 8, 28)

    def test_date_comes_from_the_ticker_not_the_station(self):
        # The event date is in the ticker, so it resolves even for a series we
        # have no station for; get_forecast_highs is what rejects those.
        assert wf.target_date(contract(series_ticker="KXNOPE")) == date(2026, 8, 28)

    def test_undated_ticker_on_unknown_series_returns_none(self):
        assert wf.target_date(
            contract(event_ticker="NOPE", series_ticker="KXNOPE")
        ) is None


class TestParsePeriods:
    def test_takes_daytime_highs_only(self):
        station = wf.get_station("KXHIGHNY")
        highs = wf._parse_periods(FORECAST_PAYLOAD, station)
        assert highs == {date(2026, 8, 27): 84.0, date(2026, 8, 28): 81.0}

    def test_converts_celsius(self):
        station = wf.get_station("KXHIGHNY")
        payload = {"properties": {"periods": [
            {"startTime": "2026-08-27T06:00:00-04:00", "isDaytime": True,
             "temperature": 30, "temperatureUnit": "C"},
        ]}}
        assert wf._parse_periods(payload, station)[date(2026, 8, 27)] == pytest.approx(86.0)

    def test_skips_periods_missing_data(self):
        station = wf.get_station("KXHIGHNY")
        payload = {"properties": {"periods": [
            {"startTime": "2026-08-27T06:00:00-04:00", "isDaytime": True,
             "temperature": None, "temperatureUnit": "F"},
            {"isDaytime": True, "temperature": 80, "temperatureUnit": "F"},
        ]}}
        assert wf._parse_periods(payload, station) == {}

    def test_empty_payload(self):
        assert wf._parse_periods({}, wf.get_station("KXHIGHNY")) == {}


class TestGetForecastHighs:
    def test_maps_event_to_its_days_high(self, monkeypatch):
        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD})
        highs = wf.get_forecast_highs([contract()])
        assert highs == {"KXHIGHNY-26AUG28": 81.0}

    def test_event_without_a_forecast_day_is_absent(self, monkeypatch):
        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD})
        far = contract(event_ticker="KXHIGHNY-26SEP15",
                       close_time=datetime(2026, 9, 15, 23, 59))
        assert wf.get_forecast_highs([far]) == {}

    def test_unknown_series_skipped(self, monkeypatch):
        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD})
        assert wf.get_forecast_highs([contract(series_ticker="KXNOPE")]) == {}

    def test_one_lookup_per_series_not_per_contract(self, monkeypatch):
        calls: list[str] = []
        install_json(
            monkeypatch,
            {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD},
            calls,
        )
        buckets = [
            contract(ticker=f"KXHIGHNY-26AUG28-B{n}") for n in (78, 80, 82, 84)
        ]
        wf.get_forecast_highs(buckets)
        assert len([c for c in calls if "/forecast" in c]) == 1

    def test_forecast_is_cached_across_calls(self, monkeypatch):
        calls: list[str] = []
        install_json(
            monkeypatch,
            {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD},
            calls,
        )
        wf.get_forecast_highs([contract()])
        wf.get_forecast_highs([contract()])
        assert len([c for c in calls if "/forecast" in c]) == 1

    def test_network_failure_returns_empty_and_trips_breaker(self, monkeypatch):
        def boom(url, params=None):
            raise RuntimeError("nws down")

        monkeypatch.setattr(wf, "_get_json", boom)
        assert wf.get_forecast_highs([contract()]) == {}
        assert not wf._available()

    def test_open_breaker_skips_requests(self, monkeypatch):
        monkeypatch.setattr(wf, "_get_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        wf.get_forecast_highs([contract()])

        calls: list[str] = []
        install_json(
            monkeypatch,
            {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD},
            calls,
        )
        assert wf.get_forecast_highs([contract()]) == {}
        assert calls == []

    def test_feeds_the_probability_model(self, monkeypatch):
        from src.analysis.event_model import build_candidates

        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD})
        contracts = [contract()]
        highs = wf.get_forecast_highs(contracts)
        candidates = build_candidates(contracts, highs, now=NOW, normalize=False)
        assert len(candidates) == 1
        assert candidates[0].forecast_high == 81.0


class TestForecastDiscussion:
    def test_returns_product_text(self, monkeypatch):
        install_json(monkeypatch, {
            "/points/": POINTS_PAYLOAD,
            "/products/types/AFD/locations/OKX": {"@graph": [{"id": "ABC123"}]},
            "/products/ABC123": {"productText": "Uncertainty in tomorrow's high."},
        })
        assert "Uncertainty" in wf.fetch_forecast_discussion("KXHIGHNY")

    def test_truncated_to_max_chars(self, monkeypatch):
        install_json(monkeypatch, {
            "/points/": POINTS_PAYLOAD,
            "/products/types/AFD/locations/OKX": {"@graph": [{"id": "ABC123"}]},
            "/products/ABC123": {"productText": "x" * 5000},
        })
        assert len(wf.fetch_forecast_discussion("KXHIGHNY", max_chars=100)) == 100

    def test_no_products_returns_empty(self, monkeypatch):
        install_json(monkeypatch, {
            "/points/": POINTS_PAYLOAD,
            "/products/types/AFD/locations/OKX": {"@graph": []},
        })
        assert wf.fetch_forecast_discussion("KXHIGHNY") == ""

    def test_failure_is_not_fatal(self, monkeypatch):
        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD})
        assert wf.fetch_forecast_discussion("KXHIGHNY") == ""

    def test_unknown_series_returns_empty(self):
        assert wf.fetch_forecast_discussion("KXNOPE") == ""


class TestTargetDateFromTicker:
    """A market stays open past the day it covers — settlement waits on the next
    morning's climate report — so close_time is not the event date."""

    def test_uses_the_date_in_the_event_ticker(self):
        c = contract(event_ticker="KXHIGHDEN-26AUG27", series_ticker="KXHIGHDEN",
                     close_time=datetime(2026, 8, 28, 18, 0))
        assert wf.target_date(c) == date(2026, 8, 27)

    def test_close_time_on_the_following_day_does_not_shift_it(self):
        # The exact live case: an AUG27 contract whose close_time lands on the
        # 28th was being priced against the 28th's forecast.
        early = contract(event_ticker="KXHIGHDEN-26AUG27", series_ticker="KXHIGHDEN",
                         close_time=datetime(2026, 8, 28, 6, 0))
        late = contract(event_ticker="KXHIGHDEN-26AUG27", series_ticker="KXHIGHDEN",
                        close_time=datetime(2026, 8, 29, 14, 0))
        assert wf.target_date(early) == wf.target_date(late) == date(2026, 8, 27)

    def test_parses_every_month(self):
        for code, month in (("JAN", 1), ("JUN", 6), ("SEP", 9), ("DEC", 12)):
            c = contract(event_ticker=f"KXHIGHNY-26{code}05")
            assert wf.target_date(c) == date(2026, month, 5)

    def test_falls_back_to_close_time_without_a_dated_ticker(self):
        c = contract(event_ticker="KXHIGHNY-SPECIAL", series_ticker="KXHIGHNY",
                     close_time=datetime(2026, 8, 28, 23, 59))
        assert wf.target_date(c) == date(2026, 8, 28)

    def test_unknown_series_without_a_dated_ticker_is_none(self):
        c = contract(event_ticker="NOPE", series_ticker="KXNOPE")
        assert wf.target_date(c) is None

    def test_impossible_date_falls_back(self):
        c = contract(event_ticker="KXHIGHNY-26FEB30", series_ticker="KXHIGHNY",
                     close_time=datetime(2026, 8, 28, 23, 59))
        assert wf.target_date(c) == date(2026, 8, 28)

    def test_yesterdays_contract_gets_no_forecast(self, monkeypatch):
        # NWS only forecasts forward, so a past event day is simply absent —
        # which is the behaviour that stops it being priced at all.
        install_json(monkeypatch, {"/points/": POINTS_PAYLOAD, "/forecast": FORECAST_PAYLOAD})
        past = contract(event_ticker="KXHIGHNY-26AUG26", series_ticker="KXHIGHNY",
                        close_time=datetime(2026, 8, 27, 5, 0))
        assert wf.get_forecast_highs([past]) == {}
