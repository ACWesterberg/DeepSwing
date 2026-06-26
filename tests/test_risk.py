from __future__ import annotations

import pytest

from src.agent.risk import RiskValidation, compute_position_size, validate_trade
from src.analysis.technical import TechnicalSignals


def _make_signals(
    ticker: str = "AAPL",
    current_price: float = 100.0,
    atr_14: float = 2.0,
) -> TechnicalSignals:
    return TechnicalSignals(
        ticker=ticker,
        ema_21=98.0,
        sma_50=95.0,
        sma_200=90.0,
        price_above_50sma=True,
        price_above_200sma=True,
        ema_21_above_50sma=True,
        atr_14=atr_14,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_pct_b=0.5,
        rsi_14=55.0,
        parabolic_sar=97.0,
        sar_is_bearish=False,
        ease_of_movement=0.01,
        obv=1_000_000.0,
        volume_ratio=1.8,
        volume_spike=True,
        fib_38_2=97.0,
        fib_61_8=94.0,
        current_price=current_price,
        current_volume=500_000.0,
    )


VALID_ENTRY = 100.0
VALID_STOP = 97.0    # 3 SEK risk, within 1.5×ATR=3.0
VALID_TARGET = 110.0  # 10 SEK reward → RRR 3.33
EQUITY = 100_000.0


class TestValidateTrade:
    def test_basic_valid_trade_approved(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is True
        assert result.quantity > 0
        assert result.rrr == pytest.approx(3.33, abs=0.005)

    def test_sell_action_always_approved(self):
        result = validate_trade(
            action="SELL",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is True
        assert result.quantity == 0.0

    def test_stop_above_entry_rejected(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=101.0,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is False
        assert "Stop loss" in result.rejection_reason

    def test_target_below_entry_rejected(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=99.0,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is False
        assert "Target" in result.rejection_reason

    def test_rrr_below_minimum_rejected(self):
        # RRR = (101 - 100) / (100 - 97) = 0.33 < 2.0
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=101.0,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is False
        assert "RRR" in result.rejection_reason

    def test_duplicate_ticker_rejected(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[{"ticker": "AAPL"}],
            signals=_make_signals(ticker="AAPL"),
        )
        assert result.approved is False
        assert "already open" in result.rejection_reason

    def test_stop_too_far_below_atr_rejected(self):
        # ATR=2.0, atr_stop = 100 - 1.5×2 = 97.0; 90% threshold = 87.3
        # stop=80 is way below 87.3 → rejected
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=80.0,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(atr_14=2.0),
        )
        assert result.approved is False
        assert "ATR" in result.rejection_reason

    def test_drawdown_mode_halves_quantity(self):
        normal = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
            is_drawdown_mode=False,
        )
        drawdown = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
            is_drawdown_mode=True,
        )
        assert drawdown.approved is True
        assert drawdown.quantity == pytest.approx(normal.quantity * 0.5, rel=1e-6)
        assert drawdown.risk_amount == pytest.approx(normal.risk_amount * 0.5, rel=1e-6)

    def test_position_sizing_1_pct_risk(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is True
        # 1% of 100_000 = 1_000 SEK at risk; risk per share = 3.0 → qty = 333.33
        expected_qty = (EQUITY * 0.01) / (VALID_ENTRY - VALID_STOP)
        assert result.quantity == pytest.approx(expected_qty, rel=1e-3)


    def test_sector_concentration_rejected_at_limit(self):
        open_positions = [
            {"ticker": "MSFT", "sector": "Technology"},
            {"ticker": "GOOGL", "sector": "Technology"},
        ]
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=open_positions,
            signals=_make_signals(ticker="NVDA"),
            candidate_sector="Technology",
        )
        assert result.approved is False
        assert "Sector" in result.rejection_reason or "sector" in result.rejection_reason

    def test_sector_concentration_allowed_below_limit(self):
        open_positions = [{"ticker": "MSFT", "sector": "Technology"}]
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=open_positions,
            signals=_make_signals(ticker="NVDA"),
            candidate_sector="Technology",
        )
        assert result.approved is True

    def test_unknown_sector_skips_concentration_check(self):
        open_positions = [
            {"ticker": "X", "sector": "Unknown"},
            {"ticker": "Y", "sector": "Unknown"},
        ]
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=VALID_STOP,
            target=VALID_TARGET,
            portfolio_equity=EQUITY,
            open_positions=open_positions,
            signals=_make_signals(ticker="Z"),
            candidate_sector="Unknown",
        )
        assert result.approved is True


class TestComputePositionSize:
    def test_basic_sizing(self):
        qty = compute_position_size(entry_price=100.0, stop_loss=97.0, portfolio_equity=100_000.0)
        assert qty == pytest.approx(100_000.0 * 0.01 / 3.0, rel=1e-6)

    def test_zero_risk_per_share_returns_zero(self):
        qty = compute_position_size(entry_price=100.0, stop_loss=100.0, portfolio_equity=100_000.0)
        assert qty == 0.0

    def test_stop_above_entry_returns_zero(self):
        qty = compute_position_size(entry_price=100.0, stop_loss=105.0, portfolio_equity=100_000.0)
        assert qty == 0.0
