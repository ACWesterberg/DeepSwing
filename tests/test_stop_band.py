from __future__ import annotations

from src.agent.risk import validate_trade
from tests.test_risk import _make_signals


def _validate(stop: float, target: float = 115.0):
    return validate_trade(
        action="BUY", entry_price=100.0, stop_loss=stop, target=target,
        portfolio_equity=100_000.0, open_positions=[],
        signals=_make_signals(current_price=100.0, atr_14=2.0),
    )


class TestDefaultStopBand:
    """Live stops sat at a median 0.8×ATR and 19 of 48 stop-outs came within a day."""

    def test_stop_inside_daily_noise_is_rejected(self):
        result = _validate(stop=98.0)  # 1×ATR
        assert not result.approved
        assert "floor" in result.rejection_reason

    def test_two_atr_stop_is_accepted(self):
        assert _validate(stop=96.0).approved  # 2×ATR, RRR 3.75

    def test_stop_beyond_ceiling_is_rejected(self):
        assert not _validate(stop=93.0, target=130.0).approved  # 3.5×ATR
