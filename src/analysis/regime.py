from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeType = Literal["trending", "mean-reverting", "neutral"]


@dataclass
class RegimeResult:
    regime: RegimeType
    hurst_exponent: float
    autocorrelation: float
    recommended_tactic: str

    def to_prompt_str(self) -> str:
        return (
            f"Regime: {self.regime} | "
            f"Hurst: {self.hurst_exponent:.3f} | "
            f"Autocorrelation(lag-1): {self.autocorrelation:.3f}\n"
            f"Recommended tactic: {self.recommended_tactic}"
        )


def classify_regime(df: pd.DataFrame, window: int = 100) -> RegimeResult:
    """
    Classify market regime using Hurst Exponent and lag-1 autocorrelation.
    Uses the last `window` close prices.
    """
    closes = df["Close"].dropna().values[-window:]
    if len(closes) < 20:
        return RegimeResult(
            regime="neutral",
            hurst_exponent=0.5,
            autocorrelation=0.0,
            recommended_tactic="Insufficient data — skip or use conservative sizing",
        )

    hurst = _hurst_exponent(closes)
    autocorr = _autocorrelation(closes, lag=1)

    if hurst > 0.55:
        regime: RegimeType = "trending"
        tactic = "EMA crossover / breakout entries; trail stops with Parabolic SAR"
    elif hurst < 0.45:
        regime = "mean-reverting"
        tactic = "Bollinger Band bounce entries; tighter targets near mean"
    else:
        regime = "neutral"
        tactic = "Reduce position size; wait for clearer regime before entering"

    return RegimeResult(
        regime=regime,
        hurst_exponent=hurst,
        autocorrelation=autocorr,
        recommended_tactic=tactic,
    )


def _hurst_exponent(ts: np.ndarray) -> float:
    """
    Estimate Hurst Exponent via Rescaled Range (R/S) analysis.
    H ≈ 0.5 = random walk; H > 0.5 = trending; H < 0.5 = mean-reverting.
    """
    n = len(ts)
    lags = []
    rs_values = []

    for lag in range(10, n // 2, max(1, n // 20)):
        sub = ts[:lag]
        mean = np.mean(sub)
        deviations = np.cumsum(sub - mean)
        r = np.max(deviations) - np.min(deviations)
        s = np.std(sub, ddof=1)
        if s > 0:
            lags.append(np.log(lag))
            rs_values.append(np.log(r / s))

    if len(lags) < 2:
        return 0.5

    coeffs = np.polyfit(lags, rs_values, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def _autocorrelation(ts: np.ndarray, lag: int = 1) -> float:
    """Lag-1 autocorrelation of log returns."""
    log_returns = np.diff(np.log(ts + 1e-10))
    if len(log_returns) <= lag:
        return 0.0
    x = log_returns[:-lag]
    y = log_returns[lag:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return round(corr, 4)
