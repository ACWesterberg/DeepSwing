from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from config.settings import settings
from src.analysis.event_model import (
    SIGMA_FLOOR_F,
    EventContract,
    bucket_bounds,
    bucket_probability,
    build_candidates,
    fair_probability,
    forecast_sigma,
)

NOW = datetime(2026, 8, 27, 12, 0, 0)


def make_contract(
    ticker: str = "KXHIGHNY-26AUG27-B82",
    *,
    strike_type: str = "between",
    floor_strike: float | None = 82,
    cap_strike: float | None = 83,
    yes_bid: float = 0.20,
    yes_ask: float = 0.24,
    close_time: datetime | None = None,
    open_interest: int = 500,
    event_ticker: str = "KXHIGHNY-26AUG27",
) -> EventContract:
    return EventContract(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker="KXHIGHNY",
        title="NYC high 82-83",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=(yes_bid + yes_ask) / 2,
        volume=1000,
        open_interest=open_interest,
        close_time=close_time or (NOW + timedelta(days=1)),
        strike_type=strike_type,
        floor_strike=floor_strike,
        cap_strike=cap_strike,
    )


class TestForecastSigma:
    def test_continuous_at_one_day(self):
        below = forecast_sigma(1.0 - 1e-9)
        above = forecast_sigma(1.0 + 1e-9)
        assert below == pytest.approx(above, abs=1e-6)
        assert forecast_sigma(1.0) == pytest.approx(settings.forecast_sigma_day1)

    def test_floored_at_resolution(self):
        assert forecast_sigma(0.0) == SIGMA_FLOOR_F
        assert forecast_sigma(-5.0) == SIGMA_FLOOR_F

    def test_monotonically_increasing(self):
        leads = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas = [forecast_sigma(x) for x in leads]
        assert sigmas == sorted(sigmas)

    def test_widens_linearly_beyond_a_day(self):
        step = forecast_sigma(5.0) - forecast_sigma(4.0)
        assert step == pytest.approx(settings.forecast_sigma_per_day)


class TestBucketBounds:
    def test_integer_between_gets_continuity_correction(self):
        # "82 to 83" means the high is 82 or 83 — an interval of width 2, not 1.
        lo, hi = bucket_bounds(make_contract(floor_strike=82, cap_strike=83))
        assert (lo, hi) == (81.5, 83.5)

    def test_half_degree_strikes_are_left_alone(self):
        lo, hi = bucket_bounds(make_contract(floor_strike=81.5, cap_strike=83.5))
        assert (lo, hi) == (81.5, 83.5)

    def test_greater_is_exclusive(self):
        # high > 85 means 86 and up
        lo, hi = bucket_bounds(
            make_contract(strike_type="greater", floor_strike=85, cap_strike=None)
        )
        assert (lo, hi) == (85.5, math.inf)

    def test_greater_or_equal_is_inclusive(self):
        lo, hi = bucket_bounds(
            make_contract(strike_type="greater_or_equal", floor_strike=85, cap_strike=None)
        )
        assert (lo, hi) == (84.5, math.inf)

    def test_less_is_exclusive(self):
        # high < 70 means 69 and below
        lo, hi = bucket_bounds(
            make_contract(strike_type="less", floor_strike=None, cap_strike=70)
        )
        assert (lo, hi) == (-math.inf, 69.5)

    def test_less_or_equal_is_inclusive(self):
        lo, hi = bucket_bounds(
            make_contract(strike_type="less_or_equal", floor_strike=None, cap_strike=70)
        )
        assert (lo, hi) == (-math.inf, 70.5)

    def test_unusable_strike_raises(self):
        with pytest.raises(ValueError):
            bucket_bounds(
                make_contract(strike_type="between", floor_strike=None, cap_strike=None)
            )


class TestBucketProbability:
    def test_full_support_is_one(self):
        assert bucket_probability(-math.inf, math.inf, 80.0, 3.0) == pytest.approx(1.0)

    def test_symmetric_around_mean(self):
        left = bucket_probability(-math.inf, 80.0, 80.0, 3.0)
        right = bucket_probability(80.0, math.inf, 80.0, 3.0)
        assert left == pytest.approx(0.5)
        assert right == pytest.approx(0.5)

    def test_one_sigma_band(self):
        assert bucket_probability(77.0, 83.0, 80.0, 3.0) == pytest.approx(0.6827, abs=1e-3)

    def test_degenerate_sigma_is_indicator(self):
        assert bucket_probability(81.5, 83.5, 82.0, 0.0) == 1.0
        assert bucket_probability(81.5, 83.5, 90.0, 0.0) == 0.0

    def test_never_outside_zero_one(self):
        for mu in (-100.0, 0.0, 80.0, 500.0):
            p = bucket_probability(81.5, 83.5, mu, 2.5)
            assert 0.0 <= p <= 1.0


class TestFairProbability:
    def test_matches_hand_computed_normal(self):
        contract = make_contract(floor_strike=82, cap_strike=82)  # exactly 82
        prob, sigma, lead_days = fair_probability(contract, forecast_high=82.0, now=NOW)
        assert lead_days == pytest.approx(1.0)
        assert sigma == pytest.approx(settings.forecast_sigma_day1)
        expected = bucket_probability(81.5, 82.5, 82.0, sigma)
        assert prob == pytest.approx(expected)

    def test_continuity_correction_roughly_doubles_a_two_degree_bucket(self):
        one_deg = fair_probability(
            make_contract(floor_strike=82, cap_strike=82), 82.0, NOW
        )[0]
        two_deg = fair_probability(
            make_contract(floor_strike=82, cap_strike=83), 82.0, NOW
        )[0]
        assert two_deg > one_deg
        assert two_deg / one_deg == pytest.approx(1.9, abs=0.2)

    def test_far_forecast_gives_near_zero(self):
        prob, _, _ = fair_probability(
            make_contract(floor_strike=82, cap_strike=83), forecast_high=60.0, now=NOW
        )
        assert prob < 1e-6

    def test_lead_time_never_negative(self):
        past = make_contract(close_time=NOW - timedelta(days=3))
        _, sigma, lead_days = fair_probability(past, 82.0, NOW)
        assert lead_days == 0.0
        assert sigma == SIGMA_FLOOR_F


def _ladder(event_ticker: str = "KXHIGHNY-26AUG27") -> list[EventContract]:
    """A complete, mutually exclusive bucket ladder for one event."""
    contracts = [
        make_contract(
            ticker=f"{event_ticker}-T74",
            strike_type="less_or_equal", floor_strike=None, cap_strike=74,
            event_ticker=event_ticker,
        )
    ]
    for low in range(75, 87, 2):
        contracts.append(
            make_contract(
                ticker=f"{event_ticker}-B{low}",
                strike_type="between", floor_strike=low, cap_strike=low + 1,
                event_ticker=event_ticker,
            )
        )
    contracts.append(
        make_contract(
            ticker=f"{event_ticker}-T87",
            strike_type="greater_or_equal", floor_strike=87, cap_strike=None,
            event_ticker=event_ticker,
        )
    )
    return contracts


class TestBuildCandidates:
    def test_complete_ladder_sums_to_one(self):
        contracts = _ladder()
        candidates = build_candidates(
            contracts, {"KXHIGHNY-26AUG27": 81.0}, now=NOW, normalize=True
        )
        assert len(candidates) == len(contracts)
        assert sum(c.fair_prob for c in candidates) == pytest.approx(1.0)

    def test_ladder_is_near_one_even_before_normalising(self):
        # The buckets partition the real line, so the raw model already sums to 1.
        candidates = build_candidates(
            _ladder(), {"KXHIGHNY-26AUG27": 81.0}, now=NOW, normalize=False
        )
        assert sum(c.fair_prob for c in candidates) == pytest.approx(1.0, abs=1e-9)

    def test_partial_ladder_is_not_rescaled(self):
        # Two buckets out of a ladder must not be inflated to sum to 1.
        partial = _ladder()[:2]
        candidates = build_candidates(
            partial, {"KXHIGHNY-26AUG27": 81.0}, now=NOW, normalize=True
        )
        assert sum(c.fair_prob for c in candidates) < 0.5

    def test_edge_is_fair_minus_ask(self):
        contract = make_contract(yes_bid=0.10, yes_ask=0.15)
        candidates = build_candidates(
            [contract], {"KXHIGHNY-26AUG27": 82.0}, now=NOW, normalize=False
        )
        c = candidates[0]
        assert c.market_prob == 0.15
        assert c.edge == pytest.approx(c.fair_prob - 0.15)

    def test_edge_recomputed_after_normalisation(self):
        candidates = build_candidates(
            _ladder(), {"KXHIGHNY-26AUG27": 81.0}, now=NOW, normalize=True
        )
        for c in candidates:
            assert c.edge == pytest.approx(c.fair_prob - c.market_prob)

    def test_contract_without_forecast_is_skipped(self):
        candidates = build_candidates([make_contract()], {}, now=NOW)
        assert candidates == []

    def test_unusable_contract_is_skipped_not_fatal(self):
        bad = make_contract(ticker="BAD", floor_strike=None, cap_strike=None)
        good = make_contract(ticker="GOOD")
        candidates = build_candidates(
            [bad, good], {"KXHIGHNY-26AUG27": 82.0}, now=NOW, normalize=False
        )
        assert [c.ticker for c in candidates] == ["GOOD"]


class TestPromptString:
    def test_states_model_probability_not_a_question(self):
        candidates = build_candidates(
            [make_contract()], {"KXHIGHNY-26AUG27": 82.0}, now=NOW, normalize=False
        )
        text = candidates[0].to_prompt_str()
        assert "Model fair probability:" in text
        assert "Edge vs ask:" in text
        assert "NWS forecast high:" in text

    def test_open_bucket_renders_infinity(self):
        contract = make_contract(
            strike_type="greater_or_equal", floor_strike=87, cap_strike=None
        )
        candidates = build_candidates(
            [contract], {"KXHIGHNY-26AUG27": 82.0}, now=NOW, normalize=False
        )
        assert "+inf" in candidates[0].to_prompt_str()
