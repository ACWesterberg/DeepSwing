from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HV_WINDOW = 30
_LOOKBACK_DAYS = 252
_TRADING_DAYS = 252


@dataclass
class VolContext:
    hv_30: float            # current 30-day realized vol, annualized
    hv_percentile: float    # 0-100: where hv_30 sits in the past year's HV values
    hv_min: float
    hv_max: float
    atm_iv: float
    iv_hv_ratio: float      # ATM IV / realized — how expensive options are vs actual movement
    iv_rank_proxy: float    # 0-100: ATM IV positioned in the year's realized-vol range

    def pricing_label(self) -> str:
        if self.iv_hv_ratio < 1.1:
            return "cheap vs realized"
        if self.iv_hv_ratio <= 1.4:
            return "fairly priced vs realized"
        return "expensive vs realized"

    def to_prompt_str(self) -> str:
        return (
            f"30-day realized vol: {self.hv_30*100:.1f}% "
            f"({self.hv_percentile:.0f}th percentile of the past year, "
            f"range {self.hv_min*100:.0f}%-{self.hv_max*100:.0f}%). "
            f"ATM IV: {self.atm_iv*100:.1f}% = {self.iv_hv_ratio:.2f}x realized "
            f"({self.pricing_label()}); IV sits at the {self.iv_rank_proxy:.0f}th "
            f"percentile of the year's realized-vol range. "
            f"High IV means the expected move is already priced into the premium."
        )


def compute_vol_context(df: pd.DataFrame, atm_iv: float) -> Optional[VolContext]:
    """Volatility context from daily OHLCV + the shortlist's ATM IV. Free-data proxy
    for IV rank: we have no IV history, so IV is ranked against the year's realized-vol
    range instead."""
    try:
        closes = df["Close"].astype(float)
        if len(closes) < _HV_WINDOW + 20:
            return None

        log_returns = np.log(closes / closes.shift(1)).dropna()
        hv_series = (log_returns.rolling(_HV_WINDOW).std() * math.sqrt(_TRADING_DAYS)).dropna()
        if hv_series.empty:
            return None

        hv_year = hv_series.iloc[-_LOOKBACK_DAYS:]
        hv_now = float(hv_year.iloc[-1])
        if hv_now <= 0:
            return None

        hv_min = float(hv_year.min())
        hv_max = float(hv_year.max())
        hv_percentile = float((hv_year < hv_now).mean() * 100)
        vol_range = hv_max - hv_min
        iv_rank = ((atm_iv - hv_min) / vol_range * 100) if vol_range > 0 else 50.0

        return VolContext(
            hv_30=hv_now,
            hv_percentile=hv_percentile,
            hv_min=hv_min,
            hv_max=hv_max,
            atm_iv=atm_iv,
            iv_hv_ratio=atm_iv / hv_now,
            iv_rank_proxy=max(0.0, min(100.0, iv_rank)),
        )
    except Exception as exc:
        logger.warning("Vol context computation failed: %s", exc)
        return None
