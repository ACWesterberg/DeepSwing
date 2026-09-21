from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.settings import settings
from src.portfolio.daily import DailyPortfolio
from src.portfolio.simulator import commission_per_leg

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class PathOutcome:
    action: str
    r_multiple: float
    pnl_pct: float
    exit_reason: str


def evaluate_forward_path(
    window: pd.DataFrame, price: float, atr: float | None, market: str = ""
) -> PathOutcome | None:
    """Evaluate a fixed ATR plan with the same daily execution engine as backtesting."""
    if atr is None or not all(math.isfinite(v) and v > 0 for v in (price, atr)) or window.empty:
        return None
    risk = settings.atr_stop_multiplier * atr
    stop = price - risk
    if stop <= 0:
        return None
    window = window.sort_index()
    # One share with enough cash for entry costs; sizing cannot affect its R.
    portfolio = DailyPortfolio(
        price * 2, commission_rate=commission_per_leg(market), slippage=settings.simulated_slippage
    )
    portfolio.open_position(
        "counterfactual", price, stop, price + settings.min_rrr * risk, 1.0,
        window.index[0].date(), trail_distance=settings.trailing_stop_atr_multiplier * atr,
    )
    previous_close = price
    for stamp, row in window.iterrows():
        close = float(row["Close"])
        if "High" in window.columns and "Low" in window.columns:
            bar = {
                "open": float(row.get("Open", previous_close)),
                "high": float(row["High"]), "low": float(row["Low"]), "close": close,
            }
        else:
            bar = close
        portfolio.update({"counterfactual": bar}, stamp.date())
        previous_close = close
        if portfolio.closed_trades:
            break
    if not portfolio.closed_trades:
        portfolio.close_position("counterfactual", previous_close, window.index[-1].date(), "horizon")
    trade = portfolio.closed_trades[0]
    r = trade.rrr_achieved
    # A cost-covering breakeven fill can leave floating-point dust around zero.
    if abs(r) < 1e-10:
        r = 0.0
    return PathOutcome("BUY" if r > 0 else "PASS", r, trade.pnl_pct, trade.exit_reason)


def label_forward_path(
    window: pd.DataFrame, price: float, atr: float | None, market: str = ""
) -> tuple[str, float] | None:
    outcome = evaluate_forward_path(window, price, atr, market)
    return (outcome.action, outcome.r_multiple) if outcome else None
