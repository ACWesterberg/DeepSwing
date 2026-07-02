from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import ta.momentum as tam
import ta.trend as tat
import ta.volatility as tav
import ta.volume as tavol

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignals:
    ticker: str
    # Trend
    ema_21: float
    sma_50: float
    sma_200: float
    price_above_50sma: bool
    price_above_200sma: bool
    ema_21_above_50sma: bool
    # Volatility
    atr_14: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_pct_b: float          # position within Bollinger Bands (0=lower, 1=upper)
    # Momentum
    rsi_14: float
    parabolic_sar: float
    sar_is_bearish: bool     # True = SAR above price (bearish)
    ease_of_movement: float
    # Volume
    obv: float
    volume_ratio: float      # current / 20-period avg
    volume_spike: bool       # True if volume_ratio >= 1.5
    # Structure
    fib_38_2: float          # 38.2% retracement level
    fib_61_8: float          # 61.8% retracement level
    # Summary
    current_price: float
    current_volume: float

    def to_prompt_str(self) -> str:
        return (
            f"Price: {self.current_price:.4f} | "
            f"21 EMA: {self.ema_21:.4f} | "
            f"50 SMA: {self.sma_50:.4f} | "
            f"200 SMA: {self.sma_200:.4f}\n"
            f"Above 50 SMA: {self.price_above_50sma} | "
            f"Above 200 SMA: {self.price_above_200sma} | "
            f"EMA21 > SMA50: {self.ema_21_above_50sma}\n"
            f"ATR(14): {self.atr_14:.4f} | "
            f"BB Upper: {self.bb_upper:.4f} | "
            f"BB Lower: {self.bb_lower:.4f} | "
            f"BB%B: {self.bb_pct_b:.2f}\n"
            f"RSI(14): {self.rsi_14:.2f} | "
            f"Parabolic SAR: {self.parabolic_sar:.4f} (bearish={self.sar_is_bearish}) | "
            f"EOM: {self.ease_of_movement:.4f}\n"
            f"OBV: {self.obv:.0f} | "
            f"Volume Ratio (vs 20-avg): {self.volume_ratio:.2f}x (spike={self.volume_spike})\n"
            f"Fib 38.2%: {self.fib_38_2:.4f} | Fib 61.8%: {self.fib_61_8:.4f}"
        )


def compute_signals(ticker: str, df: pd.DataFrame) -> Optional[TechnicalSignals]:
    """Compute all technical indicators from OHLCV DataFrame."""
    if df is None or len(df) < 210:
        logger.debug("Not enough data for %s (%d rows)", ticker, len(df) if df is not None else 0)
        return None

    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    try:
        # --- Trend ---
        ema_21_series = tat.EMAIndicator(close, window=21).ema_indicator()
        sma_50_series = tat.SMAIndicator(close, window=50).sma_indicator()
        sma_200_series = tat.SMAIndicator(close, window=200).sma_indicator()

        # --- Volatility ---
        atr_ind = tav.AverageTrueRange(high, low, close, window=14)
        atr_series = atr_ind.average_true_range()

        bb_ind = tav.BollingerBands(close, window=20, window_dev=2)
        bb_upper_series = bb_ind.bollinger_hband()
        bb_middle_series = bb_ind.bollinger_mavg()
        bb_lower_series = bb_ind.bollinger_lband()
        bb_pctb_series = bb_ind.bollinger_pband()  # 0 = lower band, 1 = upper band

        # --- Momentum ---
        rsi_series = tam.RSIIndicator(close, window=14).rsi()

        psar_ind = tat.PSARIndicator(high, low, close)
        psar_up = psar_ind.psar_up()     # bullish SAR (price is above SAR)
        psar_down = psar_ind.psar_down() # bearish SAR (price is below SAR)

        eom_series = tavol.EaseOfMovementIndicator(high, low, volume, window=14).ease_of_movement()

        # --- Volume ---
        obv_series = tavol.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # --- Extract last values ---
        price = float(close.iloc[-1])
        vol = float(volume.iloc[-1])

        ema_21 = _last(ema_21_series)
        sma_50 = _last(sma_50_series)
        sma_200 = _last(sma_200_series)
        atr_14 = _last(atr_series)
        bb_upper = _last(bb_upper_series)
        bb_middle = _last(bb_middle_series)
        bb_lower = _last(bb_lower_series)
        bb_pctb = _last(bb_pctb_series)
        rsi_14 = _last(rsi_series)
        obv = _last(obv_series)
        eom = _last(eom_series) or 0.0

        if any(v is None for v in [ema_21, sma_50, sma_200, atr_14, bb_upper, bb_lower, rsi_14, obv]):
            logger.debug("Missing indicator values for %s", ticker)
            return None

        # Parabolic SAR: psar_up is non-NaN when bullish (price above SAR), psar_down when bearish
        psar_up_val = _last(psar_up)
        psar_down_val = _last(psar_down)
        if psar_up_val is not None:
            psar_val = psar_up_val
            sar_bearish = False
        elif psar_down_val is not None:
            psar_val = psar_down_val
            sar_bearish = True
        else:
            psar_val = price
            sar_bearish = False

        # Volume ratio — measured on the last *completed* daily bar. Intraday, the
        # latest bar is still forming (partial volume), so vol/avg would read
        # ~0.1x every morning and the volume filter would reject everything until
        # near the close. Compare the previous full day to its trailing 20-day avg.
        if len(volume) >= 21:
            ref_vol = float(volume.iloc[-2])
            vol_avg_20 = float(volume.iloc[-21:-1].mean())
        else:
            ref_vol = vol
            vol_avg_20 = float(volume.iloc[-20:].mean())
        vol_ratio = ref_vol / vol_avg_20 if vol_avg_20 > 0 else 1.0

        # Fibonacci (swing high/low over last 50 bars)
        recent = df.iloc[-50:]
        swing_high = float(recent["High"].max())
        swing_low = float(recent["Low"].min())
        rng = swing_high - swing_low
        fib_38 = swing_high - 0.382 * rng
        fib_61 = swing_high - 0.618 * rng

        return TechnicalSignals(
            ticker=ticker,
            ema_21=ema_21,
            sma_50=sma_50,
            sma_200=sma_200,
            price_above_50sma=price > sma_50,
            price_above_200sma=price > sma_200,
            ema_21_above_50sma=ema_21 > sma_50,
            atr_14=atr_14,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_pct_b=bb_pctb if bb_pctb is not None else 0.5,
            rsi_14=rsi_14,
            parabolic_sar=psar_val,
            sar_is_bearish=sar_bearish,
            ease_of_movement=eom,
            obv=obv,
            volume_ratio=vol_ratio,
            volume_spike=vol_ratio >= 1.5,
            fib_38_2=fib_38,
            fib_61_8=fib_61,
            current_price=price,
            current_volume=vol,
        )

    except Exception as exc:
        logger.error("Technical analysis error for %s: %s", ticker, exc, exc_info=True)
        return None


def _last(series: pd.Series) -> Optional[float]:
    """Return last non-NaN value from a Series, or None."""
    if series is None:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)
