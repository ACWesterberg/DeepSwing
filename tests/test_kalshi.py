from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from src.data import kalshi

NOW = datetime(2026, 8, 27, 12, 0, 0)


@pytest.fixture(autouse=True)
def _clean_breaker():
    kalshi.reset_breaker()
    yield
    kalshi.reset_breaker()


def raw_market(**overrides) -> dict:
    market = {
        "ticker": "KXHIGHNY-26AUG28-B82",
        "event_ticker": "KXHIGHNY-26AUG28",
        "title": "Will the high in NYC be 82-83?",
        "yes_bid": 20,
        "yes_ask": 24,
        "last_price": 22,
        "volume": 1500,
        "open_interest": 4000,
        "close_time": "2026-08-28T23:59:00Z",
        "status": "open",
        "strike_type": "between",
        "floor_strike": 82,
        "cap_strike": 83,
    }
    market.update(overrides)
    return market


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Serves canned pages and records the params it was called with."""

    def __init__(self, pages, calls, raises=None):
        self._pages = pages
        self._calls = calls
        self._raises = raises

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None, headers=None):
        self._calls.append((url, dict(params or {})))
        if self._raises:
            raise self._raises
        page = self._pages.pop(0) if self._pages else {"markets": [], "cursor": None}
        return _FakeResponse(page)


def install_client(monkeypatch, pages, raises=None) -> list:
    calls: list = []
    monkeypatch.setattr(
        kalshi.httpx, "Client",
        lambda *a, **kw: _FakeClient(list(pages), calls, raises),
    )
    return calls


class TestParseMarket:
    def test_cents_become_probabilities(self):
        contract = kalshi._parse_market(raw_market(), "KXHIGHNY")
        assert contract.yes_bid == pytest.approx(0.20)
        assert contract.yes_ask == pytest.approx(0.24)
        assert contract.last_price == pytest.approx(0.22)

    def test_close_time_is_naive_utc(self):
        contract = kalshi._parse_market(raw_market(), "KXHIGHNY")
        assert contract.close_time == datetime(2026, 8, 28, 23, 59, 0)
        assert contract.close_time.tzinfo is None

    def test_offset_time_converted_to_utc(self):
        contract = kalshi._parse_market(
            raw_market(close_time="2026-08-28T19:59:00-04:00"), "KXHIGHNY"
        )
        assert contract.close_time == datetime(2026, 8, 28, 23, 59, 0)

    def test_strikes_and_type_carried_through(self):
        contract = kalshi._parse_market(raw_market(), "KXHIGHNY")
        assert (contract.floor_strike, contract.cap_strike) == (82.0, 83.0)
        assert contract.strike_type == "between"
        assert contract.series_ticker == "KXHIGHNY"

    def test_strike_type_lowercased(self):
        contract = kalshi._parse_market(
            raw_market(strike_type="GREATER_OR_EQUAL"), "KXHIGHNY"
        )
        assert contract.strike_type == "greater_or_equal"

    def test_one_sided_strike_is_kept(self):
        contract = kalshi._parse_market(
            raw_market(strike_type="greater", floor_strike=87, cap_strike=None),
            "KXHIGHNY",
        )
        assert contract.floor_strike == 87.0
        assert contract.cap_strike is None

    def test_event_ticker_derived_when_absent(self):
        contract = kalshi._parse_market(raw_market(event_ticker=None), "KXHIGHNY")
        assert contract.event_ticker == "KXHIGHNY-26AUG28"

    def test_missing_ticker_rejected(self):
        assert kalshi._parse_market(raw_market(ticker=None), "KXHIGHNY") is None

    def test_missing_close_time_rejected(self):
        assert kalshi._parse_market(raw_market(close_time=None), "KXHIGHNY") is None

    def test_unparseable_close_time_rejected(self):
        assert kalshi._parse_market(raw_market(close_time="soon"), "KXHIGHNY") is None

    def test_strikeless_market_rejected(self):
        assert kalshi._parse_market(
            raw_market(floor_strike=None, cap_strike=None), "KXHIGHNY"
        ) is None

    def test_absent_quotes_default_to_zero(self):
        contract = kalshi._parse_market(
            raw_market(yes_bid=None, yes_ask=None, last_price=None), "KXHIGHNY"
        )
        assert (contract.yes_bid, contract.yes_ask, contract.last_price) == (0.0, 0.0, 0.0)

    def test_prices_clamped_to_unit_interval(self):
        contract = kalshi._parse_market(raw_market(yes_ask=150, yes_bid=-10), "KXHIGHNY")
        assert contract.yes_ask == 1.0
        assert contract.yes_bid == 0.0

    def test_junk_volume_does_not_crash(self):
        contract = kalshi._parse_market(
            raw_market(volume=None, open_interest=None), "KXHIGHNY"
        )
        assert (contract.volume, contract.open_interest) == (0, 0)


class TestFetchWeatherMarkets:
    def test_returns_parsed_contracts(self, monkeypatch):
        install_client(monkeypatch, [{"markets": [raw_market()], "cursor": None}])
        contracts = kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        assert len(contracts) == 1
        assert contracts[0].ticker == "KXHIGHNY-26AUG28-B82"

    def test_filters_markets_beyond_horizon(self, monkeypatch):
        far = raw_market(ticker="FAR", close_time="2026-12-01T23:59:00Z")
        install_client(monkeypatch, [{"markets": [raw_market(), far], "cursor": None}])
        contracts = kalshi.fetch_weather_markets(
            series=["KXHIGHNY"], days_ahead=10, now=NOW
        )
        assert [c.ticker for c in contracts] == ["KXHIGHNY-26AUG28-B82"]

    def test_filters_markets_already_closed(self, monkeypatch):
        past = raw_market(ticker="PAST", close_time="2026-08-01T23:59:00Z")
        install_client(monkeypatch, [{"markets": [past], "cursor": None}])
        assert kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW) == []

    def test_requests_open_markets_for_the_series(self, monkeypatch):
        calls = install_client(monkeypatch, [{"markets": [], "cursor": None}])
        kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        _, params = calls[0]
        assert params["series_ticker"] == "KXHIGHNY"
        assert params["status"] == "open"

    def test_follows_cursor_across_pages(self, monkeypatch):
        full_page = [raw_market(ticker=f"T{i}") for i in range(kalshi._PAGE_LIMIT)]
        calls = install_client(monkeypatch, [
            {"markets": full_page, "cursor": "page2"},
            {"markets": [raw_market(ticker="LAST")], "cursor": None},
        ])
        contracts = kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        assert len(contracts) == kalshi._PAGE_LIMIT + 1
        assert calls[1][1]["cursor"] == "page2"

    def test_short_page_stops_even_with_a_repeating_cursor(self, monkeypatch):
        # Kalshi echoes a cursor on the final page; a short page must end the loop.
        calls = install_client(monkeypatch, [
            {"markets": [raw_market()], "cursor": "same"},
            {"markets": [raw_market()], "cursor": "same"},
        ])
        kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        assert len(calls) == 1

    def test_covers_every_configured_series(self, monkeypatch):
        calls = install_client(monkeypatch, [
            {"markets": [], "cursor": None}, {"markets": [], "cursor": None},
        ])
        kalshi.fetch_weather_markets(series=["KXHIGHNY", "KXHIGHCHI"], now=NOW)
        assert {params["series_ticker"] for _, params in calls} == {"KXHIGHNY", "KXHIGHCHI"}

    def test_network_failure_returns_empty_and_trips_breaker(self, monkeypatch):
        install_client(monkeypatch, [], raises=RuntimeError("connection refused"))
        assert kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW) == []
        assert not kalshi._available()

    def test_open_breaker_skips_the_request(self, monkeypatch):
        install_client(monkeypatch, [], raises=RuntimeError("down"))
        kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)

        calls = install_client(monkeypatch, [{"markets": [raw_market()], "cursor": None}])
        assert kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW) == []
        assert calls == []

    def test_reset_breaker_allows_requests_again(self, monkeypatch):
        install_client(monkeypatch, [], raises=RuntimeError("down"))
        kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        kalshi.reset_breaker()

        install_client(monkeypatch, [{"markets": [raw_market()], "cursor": None}])
        assert len(kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)) == 1

    def test_unusable_market_skipped_without_killing_the_page(self, monkeypatch):
        install_client(monkeypatch, [{
            "markets": [raw_market(ticker=None), raw_market(ticker="GOOD")],
            "cursor": None,
        }])
        contracts = kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        assert [c.ticker for c in contracts] == ["GOOD"]

    def test_contracts_feed_the_probability_model(self, monkeypatch):
        from src.analysis.event_model import build_candidates

        install_client(monkeypatch, [{"markets": [raw_market()], "cursor": None}])
        contracts = kalshi.fetch_weather_markets(series=["KXHIGHNY"], now=NOW)
        candidates = build_candidates(
            contracts, {"KXHIGHNY-26AUG28": 82.0}, now=NOW, normalize=False
        )
        assert len(candidates) == 1
        assert 0.0 < candidates[0].fair_prob < 1.0

# The exact shape the live API returned on 2026-08-28 (external-api.kalshi.com):
# prices as `*_dollars` decimal strings, open interest as `open_interest_fp`, and
# no cap_strike on a one-sided `greater` market.
LIVE_MARKET = {
    "ticker": "KXHIGHNY-26AUG28-T87",
    "event_ticker": "KXHIGHNY-26AUG28",
    "market_type": "binary",
    "strike_type": "greater",
    "floor_strike": 87,
    "close_time": "2026-08-29T05:00:00Z",
    "open_time": "2026-08-27T14:00:00Z",
    "yes_bid_dollars": "0.0000",
    "yes_ask_dollars": "0.0100",
    "no_bid_dollars": "0.9900",
    "no_ask_dollars": "1.0000",
    "last_price_dollars": "0.0100",
    "previous_price_dollars": "0.0000",
    "open_interest_fp": "1396.94",
    "liquidity_dollars": "0.0000",
    "notional_value_dollars": "1.0000",
    "no_sub_title": "88\u00b0 or above",
    "result": "",
}


class TestLiveSchema:
    def test_dollar_strings_are_not_divided_by_a_hundred(self):
        # "0.0100" is one cent as a dollar amount. Treating it as integer cents
        # would give 0.0001 and every market would look unquoted.
        contract = kalshi._parse_market(LIVE_MARKET, "KXHIGHNY")
        assert contract.yes_ask == pytest.approx(0.01)
        assert contract.last_price == pytest.approx(0.01)
        assert contract.yes_bid == 0.0

    def test_open_interest_from_the_fp_field(self):
        contract = kalshi._parse_market(LIVE_MARKET, "KXHIGHNY")
        assert contract.open_interest == 1396

    def test_missing_volume_is_not_fatal(self):
        assert kalshi._parse_market(LIVE_MARKET, "KXHIGHNY").volume == 0

    def test_one_sided_greater_market_parses(self):
        contract = kalshi._parse_market(LIVE_MARKET, "KXHIGHNY")
        assert contract.floor_strike == 87.0
        assert contract.cap_strike is None

    def test_greater_strike_matches_kalshis_own_wording(self):
        # Kalshi describes this market as "88 degrees or above"; the bucket must
        # therefore start above 87, not at it.
        from src.analysis.event_model import bucket_bounds

        lo, hi = bucket_bounds(kalshi._parse_market(LIVE_MARKET, "KXHIGHNY"))
        assert (lo, hi) == (87.5, math.inf)

    def test_legacy_integer_cents_still_parse(self):
        legacy = {**LIVE_MARKET}
        for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars",
                    "open_interest_fp"):
            legacy.pop(key)
        legacy.update({"yes_bid": 20, "yes_ask": 24, "last_price": 22,
                       "open_interest": 4000, "volume": 150})
        contract = kalshi._parse_market(legacy, "KXHIGHNY")
        assert contract.yes_ask == pytest.approx(0.24)
        assert contract.open_interest == 4000
        assert contract.volume == 150

    def test_dollar_field_wins_over_a_stale_cents_field(self):
        both = {**LIVE_MARKET, "yes_ask": 99}
        assert kalshi._parse_market(both, "KXHIGHNY").yes_ask == pytest.approx(0.01)

    def test_unquoted_market_is_rejected_by_the_screener(self):
        from src.analysis.event_model import build_candidates
        from src.analysis.event_screener import screen_event_candidates

        contract = kalshi._parse_market(
            {**LIVE_MARKET, "yes_ask_dollars": "0.0000"}, "KXHIGHNY"
        )
        candidates = build_candidates(
            [contract], {"KXHIGHNY-26AUG28": 82.0},
            now=datetime(2026, 8, 28, 12, 0), normalize=False,
        )
        assert screen_event_candidates(candidates) == []
