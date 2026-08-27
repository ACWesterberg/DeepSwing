from __future__ import annotations

from datetime import datetime

import pytest

from src.dashboard.app import _calibration, _event_position
from src.portfolio.simulator import OpenPosition

RATE = 10.0


def position(**overrides) -> OpenPosition:
    kwargs = dict(
        trade_id=1,
        ticker="KXHIGHNY-26AUG28-B82",
        market="events",
        quantity=25,
        entry_price=0.21 * RATE,
        stop_loss=0.0,
        target=RATE,
        entry_time=datetime(2026, 8, 27, 12, 0),
        current_price=0.34 * RATE,
        entry_inputs={
            "event_ticker": "KXHIGHNY-26AUG28",
            "fair_prob": 0.42,
            "ask": 0.20,
            "net_edge": 0.19,
            "forecast_high": 82.0,
            "usd_sek": RATE,
        },
    )
    kwargs.update(overrides)
    return OpenPosition(**kwargs)


class TestCalibration:
    def test_empty_record_has_no_realised_values(self):
        buckets = _calibration([])
        assert len(buckets) == 10
        assert all(b["realised"] is None and b["count"] == 0 for b in buckets)

    def test_bins_by_predicted_probability(self):
        buckets = _calibration([(0.15, 0), (0.15, 1), (0.85, 1), (0.85, 1)])
        populated = {b["bucket"]: (b["realised"], b["count"]) for b in buckets if b["count"]}
        assert populated == {"0.1-0.2": (0.5, 2), "0.8-0.9": (1.0, 2)}

    def test_certainty_lands_in_the_last_bin(self):
        # 1.0 is the upper edge of the top bin and must not fall out of every bin.
        buckets = _calibration([(1.0, 1)])
        assert buckets[-1]["count"] == 1

    def test_zero_lands_in_the_first_bin(self):
        buckets = _calibration([(0.0, 0)])
        assert buckets[0]["count"] == 1

    def test_perfectly_calibrated_model_sits_on_the_diagonal(self):
        settled = [(0.25, 1)] + [(0.25, 0)] * 3       # 25% predicted, 25% realised
        settled += [(0.75, 1)] * 3 + [(0.75, 0)]      # 75% predicted, 75% realised
        buckets = {b["bucket"]: b["realised"] for b in _calibration(settled)}
        assert buckets["0.2-0.3"] == pytest.approx(0.25)
        assert buckets["0.7-0.8"] == pytest.approx(0.75)

    def test_overconfident_model_falls_below_the_diagonal(self):
        # Claims 85%, delivers 20%.
        settled = [(0.85, 1)] + [(0.85, 0)] * 4
        realised = {b["bucket"]: b["realised"] for b in _calibration(settled)}["0.8-0.9"]
        assert realised < 0.85

    def test_bucket_midpoints_are_the_plotted_x(self):
        buckets = _calibration([])
        assert [b["predicted"] for b in buckets][:3] == [0.05, 0.15, 0.25]

    def test_custom_bin_count(self):
        assert len(_calibration([], bins=4)) == 4


class TestEventPosition:
    def test_exposes_the_model_and_market_probabilities(self):
        rendered = _event_position(position())
        assert rendered["fair_prob"] == 0.42
        assert rendered["entry_ask"] == 0.20
        assert rendered["net_edge"] == 0.19
        assert rendered["event_ticker"] == "KXHIGHNY-26AUG28"

    def test_mark_is_shown_back_in_probability_terms(self):
        # Marked at 3.40 SEK against a 10 SEK payout — a 0.34 probability.
        assert _event_position(position())["market_prob"] == pytest.approx(0.34)

    def test_contracts_mirror_quantity(self):
        assert _event_position(position())["contracts"] == 25

    def test_zero_payout_does_not_divide_by_zero(self):
        assert _event_position(position(target=0.0))["market_prob"] is None

    def test_missing_entry_inputs_are_tolerated(self):
        rendered = _event_position(position(entry_inputs={}))
        assert rendered["fair_prob"] is None
        assert rendered["event_ticker"] == ""

    def test_keeps_the_base_position_fields(self):
        rendered = _event_position(position())
        assert rendered["ticker"] == "KXHIGHNY-26AUG28-B82"
        assert "unrealised_pnl" in rendered
