from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.portfolio.simulator import Portfolio


@dataclass
class PerformanceMetrics:
    track: str
    total_trades: int
    win_rate: float
    avg_rrr: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_return_pct: float
    avg_trade_duration_days: float
    optimization_metric: float  # win_rate * avg_rrr — used by MIPRO

    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate * 100, 1),
            "avg_rrr": round(self.avg_rrr, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "avg_trade_duration_days": round(self.avg_trade_duration_days, 1),
            "optimization_metric": round(self.optimization_metric, 4),
        }


def compute_metrics(portfolio: "Portfolio") -> PerformanceMetrics:
    trades = portfolio.closed_trades

    if not trades:
        return PerformanceMetrics(
            track=portfolio.track,
            total_trades=0,
            win_rate=0.0,
            avg_rrr=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            total_return_pct=0.0,
            avg_trade_duration_days=0.0,
            optimization_metric=0.0,
        )

    returns = [t.pnl_pct for t in trades]
    winners = [r for r in returns if r > 0]
    win_rate = len(winners) / len(returns)

    rrrs = [t.rrr_achieved for t in trades if t.rrr_achieved > 0]
    avg_rrr = float(np.mean(rrrs)) if rrrs else 0.0

    durations = [t.duration_days for t in trades]
    avg_duration = float(np.mean(durations)) if durations else 0.0

    sharpe = _compute_sharpe(returns, avg_duration_days=avg_duration)

    # Max drawdown from equity curve
    equity_curve = _build_equity_curve(portfolio)
    max_dd = _max_drawdown(equity_curve)

    total_return = (portfolio.equity - portfolio.starting_equity) / portfolio.starting_equity * 100

    return PerformanceMetrics(
        track=portfolio.track,
        total_trades=len(trades),
        win_rate=win_rate,
        avg_rrr=avg_rrr,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd * 100,
        total_return_pct=total_return,
        avg_trade_duration_days=avg_duration,
        optimization_metric=win_rate * avg_rrr,
    )


def _compute_sharpe(
    returns: list[float],
    avg_duration_days: float = 1.0,
    risk_free_rate: float = 0.03,
) -> float:
    """Annualized Sharpe from per-trade returns. Trades span multiple days, so
    annualization scales by the actual average holding period — treating each
    trade as a daily return (×√252) would overstate Sharpe several-fold."""
    if len(returns) < 2:
        return 0.0
    periods_per_year = 252.0 / max(avg_duration_days, 1.0)
    arr = np.array(returns)
    excess = arr - risk_free_rate / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def _build_equity_curve(portfolio: "Portfolio") -> list[float]:
    """Reconstruct equity curve from closed trades (simplified: sequential P&L)."""
    equity = portfolio.starting_equity
    curve = [equity]
    for trade in sorted(portfolio.closed_trades, key=lambda t: t.exit_time):
        equity += trade.pnl
        curve.append(equity)
    return curve


def _max_drawdown(equity_curve: list[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve)
    rolling_max = np.maximum.accumulate(arr)
    drawdowns = (rolling_max - arr) / rolling_max
    return float(np.max(drawdowns))


def _portfolio_start_time(portfolio: "Portfolio") -> datetime:
    times = [t.entry_time for t in portfolio.closed_trades]
    times.extend(p.entry_time for p in portfolio.open_positions)
    return min(times) if times else datetime.utcnow()


def build_equity_curve_chart_data(portfolio: "Portfolio") -> list[dict]:
    """[{date: ISO8601, equity}] for the comparison chart time scale."""
    equity = portfolio.starting_equity
    points = [{"date": _portfolio_start_time(portfolio).isoformat(), "equity": equity}]
    for trade in sorted(portfolio.closed_trades, key=lambda t: t.exit_time):
        equity += trade.pnl
        points.append({
            "date": trade.exit_time.isoformat(),
            "equity": round(equity, 2),
        })
    points.append({"date": datetime.utcnow().isoformat(), "equity": round(portfolio.equity, 2)})
    return points
