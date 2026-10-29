"""Information coefficients of candidate signals over a cached OHLCV universe.

Scores a signal by how well it ranks every liquid stock's forward return on
every sampled day, not by the trades a rule happened to take. Reads the
pickles written by `backtest_rules.py --cache`:

    venv/bin/python scripts/signal_ic.py data/bt/ohlcv_nordic_2016.pkl nordic --out ic.json

--composite takes a previous run's JSON (the selection period), keeps the
signals with |t| >= 2 on 20-day returns there, and scores their equal-weight
signed rank average on this file's data — the out-of-sample test.
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

HORIZONS = (5, 20, 60)
SAMPLE_EVERY = 5          # trading days between cross-sections
MIN_NAMES = 30


def _panels(data: dict[str, pd.DataFrame], fx: dict[str, float]) -> dict[str, pd.DataFrame]:
    tickers = [t for t, df in data.items() if fx.get(t) and len(df) > 300]
    frame = {
        col: pd.DataFrame({t: data[t][col] for t in tickers}).sort_index().astype("float32")
        for col in ("Open", "High", "Low", "Close", "Volume")
    }
    for col in ("Open", "High", "Low", "Close"):
        frame[col] = frame[col].ffill(limit=3)
    # The union calendar gives a stock rows on days only another exchange traded
    # (Oslo open, Stockholm shut). Those carry no volume; leaving them NaN blanked
    # every rolling volume window for 20 days after each such holiday.
    frame["Volume"] = frame["Volume"].where(frame["Volume"].notna() | frame["Close"].isna(), 0.0)
    frame["fx"] = pd.Series({t: fx[t] for t in tickers}, dtype="float32")
    return frame


def _rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.where(loss > 0))


def signals(p: dict[str, pd.DataFrame]):
    """Yields (name, panel) one at a time; every value is known at day t's close."""
    c, h, lo, v = p["Close"], p["High"], p["Low"], p["Volume"]
    r = c.pct_change(fill_method=None)
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    prev = c.shift()
    atr_pct = np.maximum(np.maximum(h - lo, (h - prev).abs()), (lo - prev).abs()).rolling(14).mean() / c
    vol_ratio = v / v.rolling(20).mean()
    abn_turnover = v.rolling(20).mean() / v.rolling(252, min_periods=200).mean()
    turnover_pct = abn_turnover.rank(axis=1, pct=True)
    rsi14 = _rsi(c, 14)
    ret_1m = c / c.shift(21) - 1

    yield "mom_12_1", c.shift(21) / c.shift(252) - 1
    yield "mom_6_1", c.shift(21) / c.shift(126) - 1
    yield "ret_1m", ret_1m
    yield "ret_1w", c / c.shift(5) - 1
    yield "ret_1m_hi_turnover", ret_1m.where(turnover_pct >= 2 / 3)
    yield "ret_1m_lo_turnover", ret_1m.where(turnover_pct <= 1 / 3)
    yield "near_52w_high", c / h.rolling(252).max()
    yield "dist_sma200", c / sma200 - 1
    yield "dist_sma50", c / sma50 - 1
    yield "ema21_over_sma50", ema21 / sma50 - 1
    yield "rsi14", rsi14
    yield "rsi2", _rsi(c, 2)
    yield "ibs", (c - lo) / (h - lo).where(h > lo)
    yield "volatility_20d", r.rolling(20).std()
    yield "volatility_60d", r.rolling(60).std()
    yield "max_daily_ret_1m", r.rolling(21).max()
    yield "atr_pct", atr_pct
    yield "volume_spike", vol_ratio
    yield "abnormal_turnover", abn_turnover
    yield "dollar_volume", np.log((c * v).rolling(20).median() * p["fx"])
    yield "current_rules_pass", (
        (c > sma200) & (c > sma50) & (ema21 > sma50) & (rsi14 >= 40) & (rsi14 <= 65)
        & (vol_ratio <= 2) & (atr_pct >= 0.01) & (atr_pct <= 0.05)
    ).astype("float32").where(sma200.notna())


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - np.nanmean(a, axis=1, keepdims=True)
    b = b - np.nanmean(b, axis=1, keepdims=True)
    return np.nansum(a * b, axis=1) / np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))


def _newey_west_t(x: np.ndarray, lags: int) -> float:
    """t-stat of the mean with Bartlett-weighted autocovariances: consecutive
    cross-sections share most of their forward window, so a plain t overstates."""
    n = len(x)
    if n <= lags + 2:
        return float("nan")
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else float("nan")


def evaluate(sig: pd.DataFrame, fwd: dict[int, pd.DataFrame], liquid: pd.DataFrame, rows: np.ndarray) -> dict:
    s_all = sig.iloc[rows].where(liquid.iloc[rows])
    out = {}
    for hz, f in fwd.items():
        s = s_all.where(f.iloc[rows].notna())
        ff = f.iloc[rows].where(s.notna())
        n = s.notna().sum(axis=1).to_numpy()
        ok = n >= MIN_NAMES
        rs = s.rank(axis=1).to_numpy()[ok]
        rf = ff.rank(axis=1).to_numpy()[ok]
        ic = _rowwise_corr(rs, rf)
        ic = ic[np.isfinite(ic)]
        t = _newey_west_t(ic, lags=max(1, hz // SAMPLE_EVERY))
        # Long-only view: top quintile vs the liquid universe, winsorised per date.
        fv = ff.to_numpy()[ok]
        lo_q, hi_q = np.nanpercentile(fv, [1, 99], axis=1, keepdims=True)
        fv = np.clip(fv, lo_q, hi_q)
        pct = rs / np.nanmax(rs, axis=1, keepdims=True)
        top = np.where(pct >= 0.8, fv, np.nan)
        excess = np.nanmean(top, axis=1) - np.nanmean(fv, axis=1)
        out[hz] = {
            "ic": float(np.nanmean(ic)),
            "t": t,
            "ic_pos_share": float(np.mean(ic > 0)) if len(ic) else float("nan"),
            "top_quintile_excess_pct": float(np.nanmean(excess) * 100),
            "avg_names": float(n[ok].mean()) if ok.any() else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle")
    parser.add_argument("market", choices=["nordic", "eu", "us"])
    parser.add_argument("--out")
    parser.add_argument("--min-turnover-sek", type=float, default=10e6)
    parser.add_argument("--min-price-sek", type=float, default=10.0)
    parser.add_argument("--composite", help="selection-period JSON from an earlier run")
    args = parser.parse_args()

    from src.scheduler.scan_loop import _to_sek_price

    data = pickle.loads(Path(args.pickle).read_bytes())
    fx = {t: _to_sek_price(1.0, t, args.market, strict=True) for t in data}
    p = _panels(data, fx)
    c, o = p["Close"], p["Open"]
    turnover_sek = (c * p["Volume"]).rolling(20).median() * p["fx"]
    liquid = (turnover_sek >= args.min_turnover_sek) & (c * p["fx"] >= args.min_price_sek)
    # Entry at the next open, exit at the close h days later: what a daily-open trader can capture.
    fwd = {hz: (c.shift(-hz) / o.shift(-1) - 1).clip(-0.95, 3.0) for hz in HORIZONS}
    rows = np.arange(260, len(c) - max(HORIZONS) - 1, SAMPLE_EVERY)

    if args.composite:
        chosen = {
            name: np.sign(r["20"]["ic"])
            for name, r in json.loads(Path(args.composite).read_text())["signals"].items()
            if abs(r["20"]["t"]) >= 2 and name != "current_rules_pass"
        }
        print("composite of:", ", ".join(f"{'+' if w > 0 else '-'}{n}" for n, w in chosen.items()))
        total, count = None, None
        for name, sig in signals(p):
            if name in chosen:
                ranked = sig.where(liquid).rank(axis=1, pct=True) * chosen[name]
                total = ranked.fillna(0) if total is None else total + ranked.fillna(0)
                count = ranked.notna().astype("int16") if count is None else count + ranked.notna()
        composite = (total / count.where(count >= max(1, len(chosen) // 2))).astype("float32")
        result = evaluate(composite, fwd, liquid, rows)
        for hz, r in result.items():
            print(f"composite h={hz:>2}: IC {r['ic']:+.3f} (t {r['t']:+.1f}, positive {r['ic_pos_share']:.0%})  "
                  f"top-quintile excess {r['top_quintile_excess_pct']:+.2f}%")
        if args.out:
            Path(args.out).write_text(json.dumps({"chosen": chosen, "composite": result}, indent=1))
        return

    results = {}
    for name, sig in signals(p):
        results[name] = evaluate(sig, fwd, liquid, rows)
        r20 = results[name][20]
        print(f"{name:<22} IC20 {r20['ic']:+.3f} (t {r20['t']:+.1f})  top-quintile excess/20d {r20['top_quintile_excess_pct']:+.2f}%",
              flush=True)
    meta = {"pickle": args.pickle, "market": args.market, "start": str(c.index[rows[0]].date()),
            "end": str(c.index[rows[-1]].date()), "cross_sections": int(len(rows)), "tickers": int(c.shape[1])}
    if args.out:
        Path(args.out).write_text(json.dumps({"meta": meta, "signals": results}, indent=1))


if __name__ == "__main__":
    main()
