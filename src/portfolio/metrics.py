from __future__ import annotations

import math

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
    optimization_metric: float  # mean R-multiple across ALL trades (expectancy per unit risk)

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


def metrics_by_program(portfolio: "Portfolio") -> list[dict]:
    """Closed-trade stats grouped by the compiled program that decided the entry.

    The A/B the optimizer has never had. A compiled program is applied the moment
    MIPRO writes it, so without this the only available question is whether the
    book as a whole drifted — which confounds the program with the market it
    traded into. Grouping on the program that actually made each entry is the
    one comparison that isolates it.

    Trades made before fingerprinting, or by the uncompiled program, group under
    "baseline" — the arm every compiled version has to beat.

    Descriptive only. Programs run sequentially rather than side by side, so each
    one's trades come from a different stretch of market; read a gap here as a
    reason to look, not as a measured effect.
    """
    from src.agent.compiled_program import BASELINE

    buckets: dict[str, list] = {}
    for t in portfolio.closed_trades:
        buckets.setdefault(getattr(t, "program_hash", "") or BASELINE, []).append(t)

    out = []
    for program, trades in buckets.items():
        returns = [t.pnl_pct for t in trades]
        wins = [r for r in returns if r > 0]
        out.append({
            "program": program,
            "trades": len(trades),
            "win_rate": len(wins) / len(returns) if returns else 0.0,
            "mean_pnl_pct": sum(returns) / len(returns) if returns else 0.0,
            "total_pnl_pct": sum(returns),
        })
    return sorted(out, key=lambda b: b["mean_pnl_pct"], reverse=True)


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

    # Mean R-multiple over ALL trades — averaging only the winners (the old
    # behaviour) hid loss magnitudes entirely, inflating both the dashboard's
    # Avg RRR and the optimization metric.
    rrrs = [t.rrr_achieved for t in trades]
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
        # Expectancy per unit risk. The old win_rate × avg_rrr(winners only)
        # rewarded strategies whose losers were huge — losses never entered it.
        optimization_metric=avg_rrr,
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
    """Realized-equity curve from closed trades. trade.pnl is net of commissions,
    so the curve reconciles with cash once all positions are closed; open
    positions' unrealised P&L only appears in the live final point."""
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


# Scales the realized R-multiple before the tanh squash. Chosen so the metric
# still separates outcomes across the range a swing book actually produces
# (-1R to +5R): a 2.5R and a 5R differ by 0.12 here, where the previous
# formulation — raw pnl_pct at k=10 — put them 0.045 apart and treated a +15%
# and a +30% trade as near-identical. Optimizing for a fatter right tail
# requires a metric that can see one.
#
# Lives here rather than beside MIPRO so the offline replay harness can score a
# program without importing dspy — that is what lets the harness be validated
# with no model and no spend.
R_METRIC_SCALE = 0.35


def decision_metric(example, prediction, trace=None) -> float:
    """
    Reward a decision by the money it would have made, not just action-match.

    Each training example carries the realized R-multiple of the trade — profit
    per unit of risk taken, the unit the rest of the system already thinks in.
    If the model would BUY it "earns" that R; if it passes it earns nothing.
    Squashed to (0, 1):

        take a +1R winner  → ~0.67     take a -1R loser → ~0.33
        pass on anything   →  0.50     take a +5R winner → ~0.97

    Scoring R rather than raw return matters because position size is already
    risk-normalized: a 2% move on a tight stop and a 10% move on a wide one are
    the same trade to the book, and a metric denominated in percent would
    reward the volatile one for volatility alone.

    Note passing always scores exactly 0.5. On a trainset of mostly losers the
    do-nothing program therefore wins, and only the counterfactual "missed BUY"
    examples pull against that — which is why they are not optional.
    """
    pred_action = str(getattr(prediction, "action", "")).upper()
    r = float(getattr(example, "r_multiple", 0.0) or 0.0)
    realized = r if pred_action == "BUY" else 0.0
    return 0.5 + 0.5 * math.tanh(realized * R_METRIC_SCALE)
