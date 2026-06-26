from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import settings
from src.analysis.technical import TechnicalSignals

logger = logging.getLogger(__name__)


@dataclass
class RiskValidation:
    approved: bool
    quantity: float
    risk_amount: float    # SEK at risk
    rrr: float
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "quantity": self.quantity,
            "risk_amount": self.risk_amount,
            "rrr": self.rrr,
            "rejection_reason": self.rejection_reason,
        }


def validate_trade(
    action: str,
    entry_price: float,
    stop_loss: float,
    target: float,
    portfolio_equity: float,
    open_positions: list[dict],
    signals: TechnicalSignals,
    is_drawdown_mode: bool = False,
) -> RiskValidation:
    """
    Validate a proposed trade against all risk rules.
    Returns approved=True with computed quantity, or approved=False with rejection reason.
    """
    if action != "BUY":
        # SELL/HOLD don't need position sizing validation
        return RiskValidation(approved=True, quantity=0.0, risk_amount=0.0, rrr=0.0)

    if stop_loss >= entry_price:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=0.0,
            rejection_reason=f"Stop loss {stop_loss:.4f} must be below entry {entry_price:.4f}",
        )
    if target <= entry_price:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=0.0,
            rejection_reason=f"Target {target:.4f} must be above entry {entry_price:.4f}",
        )

    # Check stop-loss is within ATR-based range
    atr_stop = entry_price - settings.atr_stop_multiplier * signals.atr_14
    if stop_loss < atr_stop * 0.90:
        # Allow 10% slack below ATR stop — too tight a stop is fine, too loose is not
        suggested = entry_price - settings.atr_stop_multiplier * signals.atr_14
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=0.0,
            rejection_reason=(
                f"Stop loss {stop_loss:.4f} is too far below ATR-based stop {suggested:.4f} "
                f"(ATR={signals.atr_14:.4f})"
            ),
        )

    # RRR check
    risk_per_share = entry_price - stop_loss
    reward_per_share = target - entry_price
    rrr = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

    if rrr < settings.min_rrr:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
            rejection_reason=f"RRR {rrr:.2f} below minimum {settings.min_rrr}",
        )

    # Position sizing: risk 1% of equity (hard cap 2%)
    max_risk_sek = portfolio_equity * settings.max_risk_per_trade
    hard_cap_sek = portfolio_equity * settings.hard_cap_risk_per_trade
    risk_amount = min(max_risk_sek, hard_cap_sek)

    quantity = risk_amount / risk_per_share
    position_value = quantity * entry_price

    # Drawdown protocol: if portfolio is down >10%, halve position size
    if is_drawdown_mode:
        quantity *= 0.5
        risk_amount *= 0.5
        logger.warning("Drawdown mode active — position size halved")

    # Correlation check (simplified: reject if same ticker already open)
    open_tickers = [p.get("ticker") for p in open_positions]
    if signals.ticker in open_tickers:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
            rejection_reason=f"Position already open for {signals.ticker}",
        )

    if quantity <= 0:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
            rejection_reason="Computed quantity is zero — insufficient portfolio equity",
        )

    return RiskValidation(
        approved=True,
        quantity=round(quantity, 4),
        risk_amount=round(risk_amount, 2),
        rrr=round(rrr, 2),
    )


def compute_position_size(
    entry_price: float,
    stop_loss: float,
    portfolio_equity: float,
) -> float:
    """Compute share quantity based on 1% risk rule."""
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return 0.0
    risk_sek = portfolio_equity * settings.max_risk_per_trade
    return risk_sek / risk_per_share


