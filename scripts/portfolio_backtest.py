"""Long-only ranked portfolio vs equal-weight liquid universe, net of costs.

Ranks liquid names by nearness to the 52-week high and low 60-day volatility
(the two signals the IC study found consistent), holds the top N equal-weight,
and rebalances every K trading days with a buffer: a holding is kept while it
ranks inside the top `buffer x N`. Decisions use day t's close; trades fill at
day t+1's open. Reads the pickles written by `backtest_rules.py --cache`:

    venv/bin/python scripts/portfolio_backtest.py data/bt/ohlcv_nordic.pkl nordic --out pf.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.signal_ic import _panels  # noqa: E402

WARMUP = 260


def scores(p: dict[str, pd.DataFrame], liquid: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Equal-weight average of cross-sectional percentile ranks, liquid names only."""
    c, h = p["Close"], p["High"]
    raw = {
        "near_52w_high": c / h.rolling(252).max(),
        "low_volatility": -c.pct_change(fill_method=None).rolling(60).std(),
    }
    total = None
    for name, w in weights.items():
        ranked = raw[name].where(liquid).rank(axis=1, pct=True) * w
        total = ranked if total is None else total + ranked
    return total / sum(weights.values())


def simulate(
    opens: pd.DataFrame, score: pd.DataFrame, eligible: pd.DataFrame, rebalance_rows: list[int],
    n: int | None, buffer: float, cost_per_side: float,
) -> tuple[pd.Series, float]:
    """Daily NAV (at opens) and average annual one-way turnover. n=None holds every eligible name."""
    nav_parts, nav = [], 1.0
    weights = pd.Series(dtype=float)       # drifted weights at the coming open
    turnover_total = 0.0
    for k, i in enumerate(rebalance_rows):
        elig = eligible.iloc[i] & score.iloc[i].notna() & opens.iloc[i + 1].notna()
        ranked = score.iloc[i][elig].sort_values(ascending=False)
        if n is None:
            hold = list(ranked.index)
        else:
            keep_zone = set(ranked.index[: int(n * buffer)])
            hold = [t for t in weights.index if t in keep_zone][:n]
            hold += [t for t in ranked.index if t not in hold][: n - len(hold)]
        if not hold:
            continue
        target = pd.Series(1.0 / len(hold), index=hold)
        traded = target.subtract(weights, fill_value=0.0).abs().sum()
        turnover_total += traded / 2
        nav *= 1 - traded * cost_per_side

        end = rebalance_rows[k + 1] + 1 if k + 1 < len(rebalance_rows) else len(opens) - 1
        path = opens.iloc[i + 1: end + 1][hold].ffill()
        rel = (path / path.iloc[0]).clip(0.05, 4.0).fillna(1.0)
        values = rel.mul(target, axis=1)
        period_nav = values.sum(axis=1)
        nav_parts.append(nav * period_nav.iloc[:-1] if k + 1 < len(rebalance_rows) else nav * period_nav)
        nav *= float(period_nav.iloc[-1])
        weights = values.iloc[-1] / period_nav.iloc[-1]
    series = pd.concat(nav_parts)
    years = (series.index[-1] - series.index[0]).days / 365.25
    return series, turnover_total / years if years > 0 else 0.0


def stats(nav: pd.Series) -> dict:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    daily = nav.pct_change().dropna()
    yearly = nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
    return {
        "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
        "vol": float(daily.std() * np.sqrt(252)),
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0,
        "max_dd": float((nav / nav.cummax() - 1).min()),
        "yearly": {int(y): float(v) for y, v in yearly.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle")
    parser.add_argument("market", choices=["nordic", "eu", "us"])
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--every", type=int, default=20, help="trading days between rebalances")
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--signals", default="near_52w_high,low_volatility")
    parser.add_argument("--min-turnover-sek", type=float, default=10e6)
    parser.add_argument("--min-price-sek", type=float, default=10.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    from config.settings import settings
    from src.scheduler.scan_loop import _to_sek_price

    data = pickle.loads(Path(args.pickle).read_bytes())
    fx = {t: _to_sek_price(1.0, t, args.market, strict=True) for t in data}
    p = _panels(data, fx)
    c = p["Close"]
    turnover_sek = (c * p["Volume"]).rolling(20).median() * p["fx"]
    history = c.notna().cumsum() >= WARMUP
    liquid = (turnover_sek >= args.min_turnover_sek) & (c * p["fx"] >= args.min_price_sek) & history
    score = scores(p, liquid, {s: 1.0 for s in args.signals.split(",")})

    cost = settings.commission_pct + settings.simulated_slippage
    if args.market in ("us", "eu"):
        cost += settings.fx_commission_pct
    rows = list(range(WARMUP, len(c) - 2, args.every))

    strat_nav, strat_turn = simulate(p["Open"], score, liquid, rows, args.top, args.buffer, cost)
    bench_nav, bench_turn = simulate(p["Open"], score.notna().astype(float).where(liquid), liquid, rows,
                                     None, 1.0, cost)
    s, b = stats(strat_nav), stats(bench_nav)
    label = f"{args.market} {Path(args.pickle).stem} top{args.top} every{args.every}d [{args.signals}]"
    print(f"=== {label}: {strat_nav.index[0]:%Y-%m-%d} → {strat_nav.index[-1]:%Y-%m-%d}, cost {cost:.2%}/side")
    for name, st, turn in (("strategy", s, strat_turn), ("equal-wt", b, bench_turn)):
        print(f"  {name:<9} CAGR {st['cagr']:+6.1%}  vol {st['vol']:5.1%}  Sharpe {st['sharpe']:.2f}  "
              f"maxDD {st['max_dd']:6.1%}  turnover {turn:4.1f}x/yr")
    print("  year      strategy  equal-wt    excess")
    for y in s["yearly"]:
        print(f"  {y}      {s['yearly'][y]:+7.1%}  {b['yearly'].get(y, float('nan')):+7.1%}  "
              f"{s['yearly'][y] - b['yearly'].get(y, float('nan')):+7.1%}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "label": label, "cost_per_side": cost,
            "strategy": {**s, "turnover": strat_turn}, "benchmark": {**b, "turnover": bench_turn},
        }, indent=1))


if __name__ == "__main__":
    main()
