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
        # Wide stop (7%) with matching ATR so the position-value cap doesn't bind
        # and the halving is directly observable.
        kwargs = dict(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=93.0,
            target=115.0,  # RRR = 15/7 ≈ 2.14
            portfolio_equity=EQUITY,
            open_positions=[],
        )
        normal = validate_trade(**kwargs, signals=_make_signals(atr_14=5.0), is_drawdown_mode=False)
        drawdown = validate_trade(**kwargs, signals=_make_signals(atr_14=5.0), is_drawdown_mode=True)
        assert normal.approved is True
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
        # 1% of 100_000 = 1_000 SEK at risk; risk per share = 3.0 → qty 333.33,
        # but position value (33_333) exceeds the 25%-of-equity cap → 250 shares.
        risk_qty = (EQUITY * 0.01) / (VALID_ENTRY - VALID_STOP)
        cap_qty = EQUITY * 0.25 / VALID_ENTRY
        assert result.quantity == pytest.approx(min(risk_qty, cap_qty), rel=1e-3)

    def test_position_value_capped_at_max_position_pct(self):
        # Very tight stop → risk-based qty would be 10% risk-per-share ⇒ huge position
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=99.0,   # 1 SEK risk/share → uncapped qty 1000 (100_000 value)
            target=102.5,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
        )
        assert result.approved is True
        assert result.quantity * VALID_ENTRY <= EQUITY * 0.25 * 1.001

    def test_position_value_capped_at_available_cash(self):
        result = validate_trade(
            action="BUY",
            entry_price=VALID_ENTRY,
            stop_loss=99.0,
            target=102.5,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=_make_signals(),
            available_cash=10_000.0,
        )
        assert result.approved is True
        # Fits inside cash with commission/slippage headroom
        assert result.quantity * VALID_ENTRY <= 10_000.0

    def test_atr_check_is_currency_safe(self):
        # US ticker: entry/stop in SEK (~10.5×) but ATR stays in USD. A stop 3%
        # below entry matches the 1.5×ATR distance and must be accepted even
        # though the raw SEK-minus-USD arithmetic would be nonsense.
        signals = _make_signals(current_price=100.0, atr_14=2.0)  # native USD
        result = validate_trade(
            action="BUY",
            entry_price=1050.0,   # SEK
            stop_loss=1018.5,     # 3% below entry, == 1.5×ATR in native terms
            target=1150.0,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=signals,
        )
        assert result.approved is True

    def test_atr_check_rejects_loose_stop_in_sek(self):
        signals = _make_signals(current_price=100.0, atr_14=2.0)
        result = validate_trade(
            action="BUY",
            entry_price=1050.0,
            stop_loss=900.0,      # ~14% below entry vs 3% ATR distance
            target=1400.0,
            portfolio_equity=EQUITY,
            open_positions=[],
            signals=signals,
        )
        assert result.approved is False
        assert "ATR" in result.rejection_reason


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
