from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.portfolio_backtest as pb
from scripts.signal_ic import _panels


def _universe(n_stocks: int = 80, n_days: int = 900, planted: bool = False, seed: int = 0,
              same_sigma: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    data = {}
    for k in range(n_stocks):
        sigma = same_sigma or rng.uniform(0.01, 0.04)
        # Calm names out-earn wild ones in arithmetic terms (log drift + sigma^2/2).
        drift = (0.001 - 0.05 * sigma) if planted else 0.0
        close = 100 * np.exp(np.cumsum(rng.normal(drift, sigma, n_days)))
        data[f"T{k}"] = pd.DataFrame({
            "Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
            "Volume": np.full(n_days, 1e6),
        }, index=idx)
    return data


def _run(data: dict, n: int = 10, cost: float = 0.0, every: int = 20):
    p = _panels(data, {t: 1.0 for t in data})
    liquid = p["Close"].notna() & (p["Close"].notna().cumsum() >= pb.WARMUP)
    score = pb.scores(p, liquid, {"near_52w_high": 1.0, "low_volatility": 1.0})
    rows = list(range(pb.WARMUP, len(p["Close"]) - 2, every))
    strat, turn = pb.simulate(p["Open"], score, liquid, rows, n, 2.0, cost)
    bench, _ = pb.simulate(p["Open"], score.notna().astype(float).where(liquid), liquid, rows, None, 1.0, cost)
    return pb.stats(strat), pb.stats(bench), turn


def test_planted_low_volatility_effect_is_captured():
    s, b, _ = _run(_universe(planted=True))
    assert s["cagr"] > b["cagr"] + 0.05


def test_no_effect_means_no_free_lunch():
    # Equal volatility everywhere, so the ranks are noise; 40 of 200 names keeps
    # chance tracking error near 2% a year.
    s, b, _ = _run(_universe(n_stocks=200, planted=False, seed=3, same_sigma=0.01), n=40)
    assert abs(s["cagr"] - b["cagr"]) < 0.05


def test_costs_scale_with_turnover():
    data = _universe(planted=True)
    free, _, turn = _run(data, cost=0.0)
    costly, _, _ = _run(data, cost=0.01)
    assert free["cagr"] > costly["cagr"]
    # one-way turnover x 2 sides x 1% each, compounded roughly
    assert free["cagr"] - costly["cagr"] == pytest.approx(turn * 2 * 0.01, rel=0.5)
    assert turn > 0


def test_holding_everything_with_flat_prices_only_loses_costs():
    idx = pd.bdate_range("2018-01-01", periods=600)
    data = {f"T{k}": pd.DataFrame({c: 50.0 for c in ("Open", "High", "Low", "Close")} | {"Volume": 1e6}, index=idx)
            for k in range(20)}
    p = _panels(data, {t: 1.0 for t in data})
    liquid = p["Close"].notna() & (p["Close"].notna().cumsum() >= pb.WARMUP)
    rows = list(range(pb.WARMUP, len(idx) - 2, 20))
    nav, _ = pb.simulate(p["Open"], liquid.astype(float).where(liquid), liquid, rows, None, 1.0, 0.002)
    assert nav.iloc[-1] == pytest.approx(1 - 0.002)
