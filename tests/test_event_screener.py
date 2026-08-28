from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.settings import settings
from src.analysis.event_model import EventCandidate, EventContract
from src.analysis.event_screener import screen_event_candidates

NOW = datetime(2026, 8, 27, 12, 0, 0)


def make_candidate(
    *,
    ticker: str = "KXHIGHNY-26AUG27-B82",
    fair_prob: float = 0.40,
    ask: float = 0.25,
    bid: float | None = None,
    open_interest: int = 5_000,
) -> EventCandidate:
    contract = EventContract(
        ticker=ticker,
        event_ticker="KXHIGHNY-26AUG27",
        series_ticker="KXHIGHNY",
        title="NYC high 82-83",
        yes_bid=ask - 0.02 if bid is None else bid,
        yes_ask=ask,
        last_price=ask,
        volume=2000,
        open_interest=open_interest,
        close_time=NOW + timedelta(days=1),
        strike_type="between",
        floor_strike=82,
        cap_strike=83,
    )
    return EventCandidate(
        contract=contract,
        fair_prob=fair_prob,
        market_prob=ask,
        edge=fair_prob - ask,
        forecast_high=82.0,
        sigma=2.5,
        lead_days=1.0,
    )


def test_passes_a_clear_edge():
    assert len(screen_event_candidates([make_candidate()])) == 1


def test_rejects_edge_below_floor():
    marginal = make_candidate(fair_prob=0.25 + settings.min_event_edge / 2, ask=0.25)
    assert screen_event_candidates([marginal]) == []


def test_rejects_negative_edge():
    assert screen_event_candidates([make_candidate(fair_prob=0.10, ask=0.25)]) == []


def test_rejects_thin_open_interest():
    thin = make_candidate(open_interest=settings.min_event_open_interest - 1)
    assert screen_event_candidates([thin]) == []


def test_rejects_wide_spread():
    wide = make_candidate(ask=0.25, bid=0.25 - settings.max_event_spread - 0.01)
    assert screen_event_candidates([wide]) == []


def test_rejects_untradeable_quote():
    assert screen_event_candidates([make_candidate(ask=0.0, bid=0.0)]) == []


def test_fee_can_be_the_only_thing_that_rejects():
    # Edge clears the floor on paper, but the fee at 50c takes more than it.
    # settings.min_event_edge is 0.07; the fee at P=0.50 is 0.0175.
    candidate = make_candidate(fair_prob=0.50 + 0.07, ask=0.50)
    on_the_line = make_candidate(fair_prob=0.50 + 0.01, ask=0.50)
    assert len(screen_event_candidates([candidate])) == 1
    assert screen_event_candidates([on_the_line]) == []


def test_ranked_by_fee_adjusted_edge():
    # Equal raw edge, but the 50c contract pays a far bigger fee than the 10c one,
    # so the cheap contract must rank first.
    cheap = make_candidate(ticker="CHEAP", fair_prob=0.10 + 0.12, ask=0.10)
    pricey = make_candidate(ticker="PRICEY", fair_prob=0.50 + 0.12, ask=0.50)
    ranked = screen_event_candidates([pricey, cheap])
    assert [c.ticker for c in ranked] == ["CHEAP", "PRICEY"]


def test_caps_at_max_candidates_per_scan():
    many = [
        make_candidate(ticker=f"T{i}", fair_prob=0.40 + i * 0.001)
        for i in range(settings.max_event_candidates_per_scan + 10)
    ]
    assert len(screen_event_candidates(many)) == settings.max_event_candidates_per_scan


def test_keeps_the_strongest_when_capped():
    many = [
        make_candidate(ticker=f"T{i}", fair_prob=0.30 + i * 0.01)
        for i in range(settings.max_event_candidates_per_scan + 5)
    ]
    kept = screen_event_candidates(many)
    assert kept[0].ticker == f"T{len(many) - 1}"


def test_empty_input():
    assert screen_event_candidates([]) == []


def test_rejection_tally_is_logged(caplog):
    with caplog.at_level("INFO"):
        screen_event_candidates([
            make_candidate(fair_prob=0.10, ask=0.25),
            make_candidate(open_interest=1),
        ])
    assert "rejected:" in caplog.text


class TestOneSidedBook:
    """A resting 1c ask with nothing bid behind it is dust, and its 0.00/0.01
    'spread' passes the spread gate — so it needs its own rejection."""

    def test_rejects_a_dust_ask_with_no_bid(self):
        # The live KXHIGHNY-26AUG28-T87 book: bid 0.00, ask 0.01, oi 1396.
        dust = make_candidate(fair_prob=0.40, ask=0.01, bid=0.0, open_interest=1396)
        assert screen_event_candidates([dust]) == []

    def test_a_dust_book_would_otherwise_pass_the_spread_gate(self):
        dust = make_candidate(fair_prob=0.40, ask=0.01, bid=0.0)
        assert dust.contract.spread <= settings.max_event_spread

    def test_accepts_a_penny_market_with_a_real_bid(self):
        quoted = make_candidate(fair_prob=0.10, ask=0.02, bid=0.01)
        assert len(screen_event_candidates([quoted])) == 1


class TestImplausibleEdge:
    """No real edge this large exists on a quoted weather market; a number above
    the cap means the model is wrong, not that the market is."""

    def test_rejects_a_near_certainty_against_a_penny_market(self):
        # The live artifact: yesterday's contract priced on today's forecast.
        absurd = make_candidate(fair_prob=1.0, ask=0.01, bid=0.01)
        assert screen_event_candidates([absurd]) == []

    def test_logs_loudly_so_it_is_not_silently_swallowed(self, caplog):
        with caplog.at_level("WARNING"):
            screen_event_candidates([make_candidate(fair_prob=1.0, ask=0.01, bid=0.01)])
        assert "IMPLAUSIBLE EDGE" in caplog.text

    def test_edge_just_under_the_cap_still_passes(self):
        ask = 0.20
        ok = make_candidate(fair_prob=ask + settings.max_plausible_edge - 0.01, ask=ask)
        assert len(screen_event_candidates([ok])) == 1

    def test_edge_just_over_the_cap_is_rejected(self):
        ask = 0.20
        bad = make_candidate(fair_prob=ask + settings.max_plausible_edge + 0.01, ask=ask)
        assert screen_event_candidates([bad]) == []

    def test_ordinary_edges_are_unaffected(self):
        assert len(screen_event_candidates([make_candidate(fair_prob=0.40, ask=0.25)])) == 1
