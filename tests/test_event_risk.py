from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from config.settings import settings
from src.agent.event_risk import (
    effective_price,
    fee_per_contract,
    kalshi_fee,
    kelly_fraction,
    validate_event_trade,
)
from src.analysis.event_model import EventCandidate, EventContract

NOW = datetime(2026, 8, 27, 12, 0, 0)


def make_candidate(
    *,
    fair_prob: float = 0.40,
    ask: float = 0.25,
    open_interest: int = 10_000,
    ticker: str = "KXHIGHNY-26AUG27-B82",
    event_ticker: str = "KXHIGHNY-26AUG27",
) -> EventCandidate:
    contract = EventContract(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker="KXHIGHNY",
        title="NYC high 82-83",
        yes_bid=max(0.0, ask - 0.02),
        yes_ask=ask,
        last_price=ask,
        volume=5000,
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


class TestKalshiFee:
    def test_hundred_contracts_at_fifty_cents(self):
        # The published reference point: $1.75 per 100 contracts at 50c.
        assert kalshi_fee(100, 0.50) == pytest.approx(1.75)

    def test_peaks_at_fifty_cents(self):
        fees = [kalshi_fee(1000, p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert max(fees) == fees[2]

    def test_symmetric_about_fifty_cents(self):
        assert kalshi_fee(1000, 0.30) == pytest.approx(kalshi_fee(1000, 0.70))

    def test_cheaper_at_the_extremes(self):
        assert kalshi_fee(1000, 0.05) < kalshi_fee(1000, 0.50)
        assert kalshi_fee(1000, 0.95) < kalshi_fee(1000, 0.50)

    def test_rounds_up_to_the_cent(self):
        # 1 contract at 50c is 1.75c raw, which bills as 2c.
        assert kalshi_fee(1, 0.50) == pytest.approx(0.02)

    def test_zero_for_no_contracts(self):
        assert kalshi_fee(0, 0.5) == 0.0
        assert kalshi_fee(-5, 0.5) == 0.0

    def test_fee_is_material_share_of_stake(self):
        # At 50c the fee is 3.5% of stake — the number that kills marginal edges.
        stake = 100 * 0.50
        assert kalshi_fee(100, 0.50) / stake == pytest.approx(0.035)


class TestEffectivePrice:
    def test_adds_unrounded_fee(self):
        assert effective_price(0.50) == pytest.approx(0.50 + 0.0175)

    def test_fee_per_contract_matches_formula(self):
        assert fee_per_contract(0.25) == pytest.approx(settings.kalshi_fee_rate * 0.25 * 0.75)

    def test_always_above_raw_price(self):
        for p in (0.01, 0.25, 0.5, 0.75, 0.99):
            assert effective_price(p) > p


class TestKellyFraction:
    def test_known_value(self):
        # f* = (p - P) / (1 - P)
        assert kelly_fraction(0.60, 0.40) == pytest.approx(0.20 / 0.60)

    def test_zero_without_edge(self):
        assert kelly_fraction(0.40, 0.40) == 0.0
        assert kelly_fraction(0.30, 0.40) == 0.0

    def test_zero_on_degenerate_price(self):
        assert kelly_fraction(0.9, 0.0) == 0.0
        assert kelly_fraction(0.9, 1.0) == 0.0

    def test_grows_with_edge(self):
        assert kelly_fraction(0.70, 0.40) > kelly_fraction(0.50, 0.40)


class TestValidateEventTrade:
    def test_approves_a_clear_edge(self):
        result = validate_event_trade(make_candidate(), 10_000.0, [])
        assert result.approved
        assert result.contracts > 0
        assert result.fee_usd > 0
        assert result.total_usd == pytest.approx(result.cost_usd + result.fee_usd)

    def test_rejects_when_fees_eat_the_edge(self):
        # 1.5 points of raw edge at 50c, where the fee alone is 1.75 points.
        result = validate_event_trade(
            make_candidate(fair_prob=0.515, ask=0.50), 10_000.0, []
        )
        assert not result.approved
        assert "after fees" in result.rejection_reason
        assert result.net_edge < 0

    def test_thin_edge_survives_at_extreme_prices(self):
        # Same 1.5 points of edge, but at 10c the fee is only 0.63 points.
        result = validate_event_trade(
            make_candidate(fair_prob=0.115, ask=0.10), 10_000.0, []
        )
        assert result.approved

    def test_rejects_duplicate_contract(self):
        candidate = make_candidate()
        open_positions = [{
            "contract_ticker": candidate.ticker,
            "event_ticker": candidate.contract.event_ticker,
            "cost_usd": 50.0,
        }]
        result = validate_event_trade(candidate, 10_000.0, open_positions)
        assert not result.approved
        assert "already open" in result.rejection_reason

    def test_rejects_at_max_positions(self):
        open_positions = [
            {"contract_ticker": f"T{i}", "event_ticker": f"E{i}", "cost_usd": 1.0}
            for i in range(settings.max_event_positions)
        ]
        result = validate_event_trade(make_candidate(), 10_000.0, open_positions)
        assert not result.approved
        assert "max_event_positions" in result.rejection_reason

    def test_position_cap_binds(self):
        # A huge edge would size well past the per-position cap.
        equity = 10_000.0
        result = validate_event_trade(
            make_candidate(fair_prob=0.95, ask=0.10), equity, []
        )
        assert result.approved
        assert result.cost_usd <= settings.max_event_position_pct * equity + 0.10

    def test_book_depth_caps_size(self):
        deep = validate_event_trade(make_candidate(open_interest=10_000), 10_000.0, [])
        thin = validate_event_trade(make_candidate(open_interest=20), 10_000.0, [])
        assert thin.approved
        assert thin.contracts < deep.contracts
        assert thin.contracts <= settings.event_book_depth_fraction * 20

    def test_event_family_exposure_cap(self):
        candidate = make_candidate()
        equity = 10_000.0
        spent = settings.max_event_family_pct * equity
        open_positions = [{
            "contract_ticker": "OTHER-BUCKET",
            "event_ticker": candidate.contract.event_ticker,
            "cost_usd": spent,
        }]
        result = validate_event_trade(candidate, equity, open_positions)
        assert not result.approved
        assert "exposure cap" in result.rejection_reason

    def test_other_events_do_not_consume_family_room(self):
        # A position on a different event leaves this event's family budget
        # untouched. It is kept small so the book-wide cap does not fire instead
        # — that cap is exercised in TestAggregateExposure.
        candidate = make_candidate()
        open_positions = [{
            "contract_ticker": "OTHER",
            "event_ticker": "KXHIGHCHI-26AUG27",
            "cost_usd": 10.0,
        }]
        assert validate_event_trade(candidate, 10_000.0, open_positions).approved

    def test_rejects_sub_contract_size(self):
        # Tiny equity cannot fund even one contract.
        result = validate_event_trade(make_candidate(), 1.0, [])
        assert not result.approved
        assert "below one contract" in result.rejection_reason

    def test_rejects_degenerate_quote(self):
        assert not validate_event_trade(make_candidate(ask=0.0), 10_000.0, []).approved
        assert not validate_event_trade(make_candidate(ask=1.0), 10_000.0, []).approved

    def test_rejects_zero_equity(self):
        assert not validate_event_trade(make_candidate(), 0.0, []).approved

    def test_fee_matches_the_billed_contract_count(self):
        candidate = make_candidate(ask=0.25)
        result = validate_event_trade(candidate, 10_000.0, [])
        assert result.fee_usd == kalshi_fee(result.contracts, 0.25)
        assert result.cost_usd == pytest.approx(result.contracts * 0.25)

    def test_bigger_edge_sizes_bigger(self):
        small = validate_event_trade(make_candidate(fair_prob=0.30, ask=0.25), 10_000.0, [])
        large = validate_event_trade(make_candidate(fair_prob=0.45, ask=0.25), 10_000.0, [])
        assert large.contracts > small.contracts


class TestAggregateExposure:
    """Event positions across different cities and days are not diversified —
    each is the same wager that forecast_sigma beats the market's spread."""

    def _positions(self, count: int, cost_each: float) -> list[dict]:
        return [
            {"contract_ticker": f"T{i}", "event_ticker": f"KXHIGH{i}-26AUG28",
             "cost_usd": cost_each}
            for i in range(count)
        ]

    def test_book_wide_cap_blocks_further_entries(self):
        equity = 10_000.0
        spent = settings.max_event_total_pct * equity
        result = validate_event_trade(make_candidate(), equity, self._positions(3, spent / 3))
        assert not result.approved
        assert "aggregate cap" in result.rejection_reason

    def test_cap_applies_across_different_events(self):
        # Each position is on its own event, so the family cap never fires; only
        # the aggregate cap stops the book concentrating.
        equity = 10_000.0
        nearly = settings.max_event_total_pct * equity * 0.99
        result = validate_event_trade(make_candidate(), equity, self._positions(5, nearly / 5))
        assert result.total_usd <= settings.max_event_total_pct * equity + 0.10

    def test_room_remaining_is_still_tradeable(self):
        equity = 10_000.0
        half = settings.max_event_total_pct * equity / 2
        assert validate_event_trade(make_candidate(), equity, self._positions(1, half)).approved

    def test_empty_book_is_unconstrained_by_the_aggregate_cap(self):
        result = validate_event_trade(make_candidate(), 10_000.0, [])
        assert result.approved
        assert result.cost_usd <= settings.max_event_position_pct * 10_000.0 + 0.10

    def test_aggregate_cap_is_tighter_than_naive_independent_sizing(self):
        # 20 positions at the 2% per-position cap would be 40% of equity.
        naive = settings.max_event_positions * settings.max_event_position_pct
        assert settings.max_event_total_pct < naive
