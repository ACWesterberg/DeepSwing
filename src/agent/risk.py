from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from config.settings import settings
from src.analysis.technical import TechnicalSignals

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_return_correlations(
    candidate_df: "pd.DataFrame",
    open_tickers: list[str],
    ohlcv_map: dict,
    window: int = 60,
) -> dict[str, float]:
    """
    Pairwise daily-return correlation between the candidate and each open
    position whose OHLCV is in ohlcv_map (the same-market batch, so no extra
    network). Cross-market positions are skipped — their bars don't align and
    the different sessions mute correlation anyway. Never raises into a scan.
    """
    correlations: dict[str, float] = {}
    if candidate_df is None or not open_tickers:
        return correlations
    try:
        candidate_returns = (
            candidate_df["Close"].dropna().tail(window + 1).pct_change().dropna()
        )
    except Exception:
        return correlations

    for ticker in open_tickers:
        df = ohlcv_map.get(ticker)
        if df is None:
            continue
        try:
            other_returns = df["Close"].dropna().tail(window + 1).pct_change().dropna()
            a, b = candidate_returns.align(other_returns, join="inner")
            if len(a) < 20:
                continue  # too little overlap for a meaningful estimate
            corr = float(a.corr(b))
            if not math.isnan(corr):
                correlations[ticker] = corr
        except Exception as exc:
            logger.debug("Correlation computation failed for %s: %s", ticker, exc)
    return correlations


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
    candidate_sector: str = "",
    available_cash: float | None = None,
    position_correlations: Optional[dict[str, float]] = None,
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

    # Check stop-loss is within ATR-based range. Compared as fractions of price so
    # the check is currency-safe: entry/stop arrive in SEK while signals.atr_14 is
    # in the ticker's native currency. Slack is 10% of the ATR distance — too
    # tight a stop is fine, too loose is not.
    if signals.current_price > 0 and signals.atr_14 > 0 and entry_price > 0:
        atr_frac = settings.atr_stop_multiplier * signals.atr_14 / signals.current_price
        stop_frac = (entry_price - stop_loss) / entry_price
        if stop_frac > atr_frac * 1.10:
            suggested = entry_price * (1 - atr_frac)
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

    # Drawdown protocol: if portfolio is down >10%, halve position size
    if is_drawdown_mode:
        quantity *= 0.5
        risk_amount *= 0.5
        logger.warning("Drawdown mode active — position size halved")

    # Cap position value at max_position_pct of equity and at available cash —
    # risk-based sizing alone is unbounded (a tight stop yields a position the
    # portfolio can't fund, which would silently fail at execution).
    position_value = quantity * entry_price
    max_value = portfolio_equity * settings.max_position_pct
    if available_cash is not None:
        # Headroom for slippage + commission applied at execution
        max_value = min(max_value, available_cash * 0.98)
    if position_value > max_value and position_value > 0:
        scale = max(max_value, 0.0) / position_value
        quantity *= scale
        risk_amount *= scale
        logger.info(
            "Position for %s scaled to %.0f%% by value cap (%.0f SEK)",
            signals.ticker, scale * 100, max_value,
        )

    # Duplicate ticker check
    open_tickers = [p.get("ticker") for p in open_positions]
    if signals.ticker in open_tickers:
        return RiskValidation(
            approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
            rejection_reason=f"Position already open for {signals.ticker}",
        )

    # Pairwise return-correlation cap — a "diversified" book of positions that
    # all move together is one big position with extra commission
    if position_correlations:
        worst_ticker, worst = max(position_correlations.items(), key=lambda kv: kv[1])
        if worst > settings.max_sector_correlation:
            return RiskValidation(
                approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
                rejection_reason=(
                    f"Return correlation {worst:.2f} with open position {worst_ticker} "
                    f"exceeds max {settings.max_sector_correlation}"
                ),
            )

    # Sector concentration check
    if candidate_sector and candidate_sector != "Unknown":
        sector_count = sum(
            1 for p in open_positions
            if p.get("sector") == candidate_sector
        )
        if sector_count >= settings.max_positions_per_sector:
            return RiskValidation(
                approved=False, quantity=0.0, risk_amount=0.0, rrr=rrr,
                rejection_reason=(
                    f"Sector '{candidate_sector}' already has {sector_count} open position(s) "
                    f"(max {settings.max_positions_per_sector})"
                ),
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


