from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.signal_ic as ic


def _universe(n_stocks: int = 60, n_days: int = 700, drift_sd: float = 0.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = {}
    for k in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(rng.normal(0, drift_sd), 0.02, n_days)))
        data[f"T{k}"] = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
            "Volume": rng.integers(200_000, 1_000_000, n_days).astype(float),
        }, index=idx)
    return data


def _setup(data: dict):
    panels = ic._panels(data, {t: 1.0 for t in data})
    c, o = panels["Close"], panels["Open"]
    fwd = {hz: (c.shift(-hz) / o.shift(-1) - 1) for hz in ic.HORIZONS}
    rows = np.arange(260, len(c) - max(ic.HORIZONS) - 1, ic.SAMPLE_EVERY)
    return panels, fwd, c.notna(), rows


def test_planted_persistence_shows_up_as_momentum_ic():
    # Each stock has its own persistent drift, so past winners keep winning.
    panels, fwd, liquid, rows = _setup(_universe(drift_sd=0.0015))
    sig = dict(ic.signals(panels))["mom_12_1"]
    result = ic.evaluate(sig, fwd, liquid, rows)
    assert result[60]["ic"] > 0.1 and result[60]["t"] > 3


def test_noise_signal_scores_near_zero():
    panels, fwd, liquid, rows = _setup(_universe())
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=panels["Close"].shape), index=panels["Close"].index,
                         columns=panels["Close"].columns)
    result = ic.evaluate(noise, fwd, liquid, rows)
    assert abs(result[20]["ic"]) < 0.03


def test_signals_never_read_the_future():
    data = _universe(n_days=400)
    cut = 330
    altered = {t: df.copy() for t, df in data.items()}
    for df in altered.values():
        df.iloc[cut + 1:, :4] *= 3.0
        df.iloc[cut + 1:, 4] *= 0.1
    a = dict(ic.signals(ic._panels(data, {t: 1.0 for t in data})))
    b = dict(ic.signals(ic._panels(altered, {t: 1.0 for t in altered})))
    for name in a:
        pd.testing.assert_frame_equal(a[name].iloc[: cut + 1], b[name].iloc[: cut + 1], check_dtype=False,
                                      obj=name)


def test_composite_skips_the_rules_flag_and_weak_signals(tmp_path, monkeypatch, capsys):
    import json
    import pickle
    import sys
    import types

    data = _universe(drift_sd=0.0015, n_stocks=50)
    pkl = tmp_path / "u.pkl"
    pkl.write_bytes(pickle.dumps(data))
    selection = tmp_path / "sel.json"
    selection.write_text(json.dumps({"signals": {
        "mom_12_1": {"20": {"ic": 0.1, "t": 4.0}},
        "rsi2": {"20": {"ic": -0.01, "t": -0.5}},
        "current_rules_pass": {"20": {"ic": 0.05, "t": 3.0}},
    }}))
    monkeypatch.setitem(sys.modules, "src.scheduler.scan_loop",
                        types.SimpleNamespace(_to_sek_price=lambda *a, **k: 1.0))
    out = tmp_path / "comp.json"
    monkeypatch.setattr(sys, "argv", ["signal_ic", str(pkl), "us", "--composite", str(selection),
                                      "--min-turnover-sek", "0", "--min-price-sek", "0", "--out", str(out)])
    ic.main()
    result = json.loads(out.read_text())
    assert result["chosen"] == {"mom_12_1": 1.0}
    assert result["composite"]["60"]["ic"] == pytest.approx(
        ic.evaluate(dict(ic.signals(ic._panels(data, {t: 1.0 for t in data})))["mom_12_1"].rank(axis=1, pct=True),
                    *_setup(data)[1:])[60]["ic"], abs=1e-6)


def test_a_holiday_on_one_exchange_does_not_blank_volume_windows():
    idx = pd.bdate_range("2021-01-01", periods=400)
    oslo = pd.DataFrame({c: 100.0 for c in ("Open", "High", "Low", "Close")} | {"Volume": 1e6}, index=idx)
    stockholm = oslo.drop(idx[330])                    # Stockholm shut on a day Oslo traded
    panels = ic._panels({"EQNR.OL": oslo, "VOLV-B.ST": stockholm}, {"EQNR.OL": 1.0, "VOLV-B.ST": 1.0})
    median_turnover = (panels["Close"] * panels["Volume"]).rolling(20).median()
    assert median_turnover["VOLV-B.ST"].iloc[331:350].notna().all()
