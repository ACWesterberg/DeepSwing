from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from config.settings import settings
from src.analysis.options_math import bs_delta, bs_price, bs_theta_per_day, intrinsic_value
from src.data.options_chain import CONTRACT_MULTIPLIER, OptionContract, _passes_filters, format_shortlist
from src.agent.options_risk import validate_option_trade
from src.portfolio.options_simulator import OptionPosition, OptionsPortfolio


def make_contract(**overrides) -> OptionContract:
    base = dict(
        contract_symbol="AAPL260821C00230000",
        underlying="AAPL",
        right="call",
        strike=230.0,
        expiry=date.today() + timedelta(days=40),
        bid=11.80,
        ask=12.20,
        last=12.00,
        volume=500,
        open_interest=5000,
        implied_vol=0.30,
        dte=40,
        mid=12.00,
        spread_pct=(12.20 - 11.80) / 12.00,
        delta=0.55,
        theta_per_day=-0.08,
    )
    base.update(overrides)
    return OptionContract(**base)


class TestOptionsMath:
    def test_call_price_matches_known_value(self):
        # S=100, K=100, 1y (365 DTE), IV=20%, r=5% → BS call ≈ 10.45
        price = bs_price(100, 100, 365, 0.20, "call", r=0.05)
        assert price == pytest.approx(10.45, abs=0.05)

    def test_put_call_parity(self):
        s, k, dte, iv, r = 105.0, 100.0, 90, 0.35, 0.04
        call = bs_price(s, k, dte, iv, "call", r)
        put = bs_price(s, k, dte, iv, "put", r)
        import math
        forward = s - k * math.exp(-r * dte / 365.0)
        assert call - put == pytest.approx(forward, abs=1e-9)

    def test_expiry_returns_intrinsic(self):
        assert bs_price(110, 100, 0, 0.3, "call") == 10.0
        assert bs_price(90, 100, 0, 0.3, "call") == 0.0
        assert bs_price(90, 100, 0, 0.3, "put") == 10.0

    def test_delta_bounds_and_direction(self):
        call_delta = bs_delta(100, 100, 40, 0.3, "call")
        put_delta = bs_delta(100, 100, 40, 0.3, "put")
        assert 0.4 < call_delta < 0.7
        assert -0.6 < put_delta < -0.3
        assert call_delta - put_delta == pytest.approx(1.0, abs=1e-9)

    def test_theta_is_negative_for_long_atm(self):
        assert bs_theta_per_day(100, 100, 40, 0.3, "call") < 0
        assert bs_theta_per_day(100, 100, 40, 0.3, "put") < 0

    def test_intrinsic_value(self):
        assert intrinsic_value(120, 100, "call") == 20
        assert intrinsic_value(80, 100, "call") == 0
        assert intrinsic_value(80, 100, "put") == 20


class TestChainFilters:
    def test_liquid_atm_contract_passes(self):
        assert _passes_filters(make_contract())

    def test_rejects_low_delta(self):
        assert not _passes_filters(make_contract(delta=0.10))

    def test_rejects_deep_itm_delta(self):
        assert not _passes_filters(make_contract(delta=0.90))

    def test_rejects_low_open_interest(self):
        assert not _passes_filters(make_contract(open_interest=10))

    def test_rejects_wide_spread(self):
        assert not _passes_filters(make_contract(spread_pct=0.20))

    def test_rejects_dead_volume(self):
        assert not _passes_filters(make_contract(volume=0))

    def test_shortlist_formatting_is_indexed(self):
        text = format_shortlist([make_contract(), make_contract(strike=240.0)])
        assert text.splitlines()[0].startswith("[0] AAPL C$230")
        assert text.splitlines()[1].startswith("[1] AAPL C$240")

    def test_rejects_unreachable_breakeven(self):
        # expected move covers only half the distance to breakeven
        assert not _passes_filters(make_contract(move_coverage=0.5))
        assert _passes_filters(make_contract(move_coverage=1.5))

    def test_prompt_line_includes_breakeven_and_expected_move(self):
        c = make_contract(
            breakeven=242.0, breakeven_move_pct=0.052,
            expected_move=18.3, move_coverage=1.6,
        )
        line = c.to_prompt_line(0)
        assert "BE $242.00 (+5.2%)" in line
        assert "exp.move ±$18.30 = 1.6x BE distance" in line


class TestExpectedMove:
    def test_parse_rows_computes_breakeven_coverage(self):
        import pandas as pd
        from src.data.options_chain import _parse_rows

        frame = pd.DataFrame([{
            "contractSymbol": "AAPL260821C00230000",
            "strike": 230.0, "bid": 11.80, "ask": 12.20, "lastPrice": 12.00,
            "volume": 500, "openInterest": 5000, "impliedVolatility": 0.30,
        }])
        expiry = date.today() + timedelta(days=36)
        [c] = _parse_rows(frame, "AAPL", "call", expiry, spot=225.0, atr=4.0)
        assert c.breakeven == pytest.approx(242.0)                    # 230 + 12 mid
        assert c.breakeven_move_pct == pytest.approx(17.0 / 225.0)
        assert c.expected_move == pytest.approx(24.0)                 # 4.0 * sqrt(36)
        assert c.move_coverage == pytest.approx(24.0 / 17.0)

    def test_parse_rows_without_atr_skips_coverage_gate(self):
        import pandas as pd
        from src.data.options_chain import _parse_rows

        frame = pd.DataFrame([{
            "contractSymbol": "AAPL260821C00230000",
            "strike": 230.0, "bid": 11.80, "ask": 12.20, "lastPrice": 12.00,
            "volume": 500, "openInterest": 5000, "impliedVolatility": 0.30,
        }])
        [c] = _parse_rows(frame, "AAPL", "call", date.today() + timedelta(days=36), spot=225.0)
        assert c.expected_move == 0.0
        assert c.move_coverage == float("inf")


class TestVolContext:
    @staticmethod
    def make_df(daily_vol: float, days: int = 300) -> "pd.DataFrame":
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(42)
        returns = rng.normal(0, daily_vol, days)
        closes = 100.0 * np.exp(np.cumsum(returns))
        return pd.DataFrame({"Close": closes})

    def test_computes_annualized_realized_vol(self):
        import math
        from src.analysis.vol_context import compute_vol_context
        ctx = compute_vol_context(self.make_df(0.02), atm_iv=0.40)
        assert ctx is not None
        expected_annual = 0.02 * math.sqrt(252)  # ≈ 0.32
        assert ctx.hv_30 == pytest.approx(expected_annual, rel=0.35)
        assert 0 <= ctx.hv_percentile <= 100
        assert 0 <= ctx.iv_rank_proxy <= 100

    def test_pricing_labels(self):
        from src.analysis.vol_context import compute_vol_context
        df = self.make_df(0.02)
        cheap = compute_vol_context(df, atm_iv=0.20)
        expensive = compute_vol_context(df, atm_iv=0.60)
        assert cheap.iv_hv_ratio < expensive.iv_hv_ratio
        assert "cheap" in cheap.pricing_label()
        assert "expensive" in expensive.pricing_label()
        assert "priced into the premium" in expensive.to_prompt_str()

    def test_too_little_data_returns_none(self):
        from src.analysis.vol_context import compute_vol_context
        assert compute_vol_context(self.make_df(0.02, days=20), atm_iv=0.3) is None


class TestOptionsRisk:
    def test_approves_and_sizes_by_premium_budget(self):
        # Premium 120 SEK/share → 12 015 SEK/contract incl. commission;
        # 1% of 2.5M = 25 000 → 2 contracts
        result = validate_option_trade(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.8,
            max_loss_pct=0.4,
            portfolio_equity=2_500_000.0,
            open_underlyings=[],
        )
        assert result.approved
        assert result.contracts == 2
        assert result.reward_risk == 2.0

    def test_single_contract_stretches_to_hard_cap(self):
        # 1 contract = ~12 015 SEK; 1% of 1M = 10 000 (too small), 2% = 20 000 → allow 1
        result = validate_option_trade(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.8,
            max_loss_pct=0.4,
            portfolio_equity=1_000_000.0,
            open_underlyings=[],
        )
        assert result.approved
        assert result.contracts == 1

    def test_rejects_above_hard_cap(self):
        result = validate_option_trade(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.8,
            max_loss_pct=0.4,
            portfolio_equity=100_000.0,  # hard cap 2 000 SEK < one contract
            open_underlyings=[],
        )
        assert not result.approved
        assert "hard cap" in result.rejection_reason

    def test_rejects_poor_reward_risk(self):
        result = validate_option_trade(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.5,
            max_loss_pct=0.4,
            portfolio_equity=2_500_000.0,
            open_underlyings=[],
        )
        assert not result.approved
        assert "Reward/risk" in result.rejection_reason

    def test_rejects_duplicate_underlying(self):
        result = validate_option_trade(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.8,
            max_loss_pct=0.4,
            portfolio_equity=2_500_000.0,
            open_underlyings=["AAPL"],
        )
        assert not result.approved
        assert "already open" in result.rejection_reason

    def test_drawdown_mode_halves_budget(self):
        kwargs = dict(
            contract=make_contract(),
            premium_sek_per_share=120.0,
            profit_target_pct=0.8,
            max_loss_pct=0.4,
            portfolio_equity=5_000_000.0,
            open_underlyings=[],
        )
        normal = validate_option_trade(**kwargs)
        halved = validate_option_trade(**kwargs, is_drawdown_mode=True)
        assert normal.contracts == 4
        assert halved.contracts == 2


def open_test_position(portfolio: OptionsPortfolio, **overrides):
    defaults = dict(
        contract_symbol="AAPL260821C00230000",
        underlying="AAPL",
        right="call",
        strike=230.0,
        expiry=date.today() + timedelta(days=40),
        contracts=2,
        entry_premium=120.0,
        profit_target_pct=0.8,
        max_loss_pct=0.4,
        time_stop_dte=7,
        regime="trending",
        reasoning="test",
        confidence=0.8,
    )
    defaults.update(overrides)
    return portfolio.open_option(**defaults)


class TestOptionsSimulator:
    def test_open_deducts_premium_and_commission(self):
        p = OptionsPortfolio("claude-opt")
        start_cash = p.cash
        position = open_test_position(p)
        assert position is not None
        cost = 120.0 * CONTRACT_MULTIPLIER * 2
        commission = settings.options_commission_per_contract_sek * 2
        assert p.cash == pytest.approx(start_cash - cost - commission)
        assert p.equity == pytest.approx(start_cash - commission)

    def test_close_realizes_pnl(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        closed = p.close_option(position.trade_id, 180.0, "profit_target")
        assert closed is not None
        assert closed.pnl == pytest.approx((180.0 - 120.0) * CONTRACT_MULTIPLIER * 2)
        assert closed.pnl_pct == pytest.approx(0.5)
        assert closed.rrr_achieved == pytest.approx(0.5 / 0.4)
        assert not p.open_positions

    def test_update_premiums_hits_profit_target(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        closed = p.update_premiums({position.contract_symbol: 120.0 * 1.8})
        assert len(closed) == 1
        assert closed[0].exit_reason == "profit_target"

    def test_update_premiums_hits_premium_stop(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        closed = p.update_premiums({position.contract_symbol: 120.0 * 0.55})
        assert len(closed) == 1
        assert closed[0].exit_reason == "premium_stop"

    def test_update_premiums_time_stop(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p, expiry=date.today() + timedelta(days=5))
        closed = p.update_premiums({position.contract_symbol: 121.0})
        assert len(closed) == 1
        assert closed[0].exit_reason == "time_stop"

    def test_update_premiums_holds_inside_bands(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        closed = p.update_premiums({position.contract_symbol: 130.0})
        assert closed == []
        assert p.open_positions[0].current_premium == 130.0

    def test_expired_positions_detected(self):
        p = OptionsPortfolio("claude-opt")
        open_test_position(p, expiry=date.today() - timedelta(days=1))
        open_test_position(
            p, underlying="MSFT", contract_symbol="MSFT260821C00500000",
            expiry=date.today() + timedelta(days=30),
        )
        expired = p.expired_positions()
        assert len(expired) == 1
        assert expired[0].underlying == "AAPL"

    def test_state_roundtrip(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        p.close_option(position.trade_id, 60.0, "premium_stop")
        open_test_position(
            p, underlying="MSFT", contract_symbol="MSFT260821C00500000",
        )
        state = p.export_state()

        restored = OptionsPortfolio("claude-opt")
        restored.import_state(state)
        assert restored.cash == pytest.approx(p.cash)
        assert restored.equity == pytest.approx(p.equity)
        assert len(restored.open_positions) == 1
        assert restored.open_positions[0].contract_symbol == "MSFT260821C00500000"
        assert len(restored.closed_trades) == 1
        assert restored.closed_trades[0].exit_reason == "premium_stop"
        assert restored._next_trade_id == p._next_trade_id

    def test_insufficient_cash_rejected(self):
        p = OptionsPortfolio("claude-opt")
        assert open_test_position(p, entry_premium=10_000.0) is None
        assert p.cash == settings.options_starting_capital_sek

    def test_position_to_dict_maps_premium_levels(self):
        p = OptionsPortfolio("claude-opt")
        position = open_test_position(p)
        d = position.to_dict()
        assert d["stop_loss"] == pytest.approx(120.0 * 0.6)
        assert d["target"] == pytest.approx(120.0 * 1.8)
        assert "AAPL C$230" in d["ticker"]
