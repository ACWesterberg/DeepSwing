from __future__ import annotations

import logging
import math
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
    # Median daily traded value (close × volume) over the last 20 completed
    # bars, in the quote unit — the screener converts it to SEK.
    avg_turnover: float = 0.0

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


def compute_signals(
    ticker: str, df: pd.DataFrame, last_bar_complete: bool = False,
) -> Optional[TechnicalSignals]:
    """Compute all technical indicators from OHLCV DataFrame.

    last_bar_complete=False treats the final row as today's still-forming bar."""
    if df is None or len(df) < 210:
        logger.debug("Not enough data for %s (%d rows)", ticker, len(df) if df is not None else 0)
        return None
    try:
        frame = compute_signal_frame(df)
        return signals_at(ticker, frame, len(frame) - 1, last_bar_complete)
    except Exception as exc:
        logger.error("Technical analysis error for %s: %s", ticker, exc, exc_info=True)
        return None


def compute_signal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Every indicator series for df, one row per bar.

    Each series is causal, so row i holds exactly what compute_signals would
    read from df[:i+1] — a backtest computes this once per ticker instead of
    once per ticker per day."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    bb_ind = tav.BollingerBands(close, window=20, window_dev=2)
    psar_up, psar_down = _parabolic_sar(high.to_numpy(float), low.to_numpy(float), close.to_numpy(float))
    vol_avg_20 = volume.rolling(20).mean()
    # Volume ratio and turnover are read from a *completed* daily bar. Intraday
    # the latest bar is still forming (partial volume), so vol/avg would read
    # ~0.1x every morning; signals_at shifts these by one bar in that case.
    vol_ratio = (volume / vol_avg_20.where(vol_avg_20 > 0)).fillna(1.0)
    turnover = (close * volume).rolling(20).median()

    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "ema_21": tat.EMAIndicator(close, window=21).ema_indicator(),
        "sma_50": tat.SMAIndicator(close, window=50).sma_indicator(),
        "sma_200": tat.SMAIndicator(close, window=200).sma_indicator(),
        "atr_14": tav.AverageTrueRange(high, low, close, window=14).average_true_range(),
        "bb_upper": bb_ind.bollinger_hband(),
        "bb_middle": bb_ind.bollinger_mavg(),
        "bb_lower": bb_ind.bollinger_lband(),
        "bb_pct_b": bb_ind.bollinger_pband(),  # 0 = lower band, 1 = upper band
        "rsi_14": tam.RSIIndicator(close, window=14).rsi(),
        "psar_up": psar_up,      # bullish SAR (price is above SAR)
        "psar_down": psar_down,  # bearish SAR (price is below SAR)
        "eom": tavol.EaseOfMovementIndicator(high, low, volume, window=14).ease_of_movement(),
        "obv": tavol.OnBalanceVolumeIndicator(close, volume).on_balance_volume(),
        "vol_ratio": vol_ratio,
        "turnover": turnover,
        # Fibonacci swing high/low over the last 50 bars
        "swing_high": high.rolling(50, min_periods=1).max(),
        "swing_low": low.rolling(50, min_periods=1).min(),
    }, index=df.index)


def _parabolic_sar(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, step: float = 0.02, max_step: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Wilder's Parabolic SAR as (up, down), NaN where the other trend holds.

    ta.trend.PSARIndicator runs the same loop through per-element pandas
    indexing — ~0.45 s a ticker, 90% of a scan's indicator time — and on a
    DatetimeIndex its `psar[i] = high2` appends a row instead of clamping."""
    h, lo = high.tolist(), low.tolist()
    n = len(h)
    psar = close.tolist()
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    up_trend = True
    af = step
    up_high, down_low = h[0], lo[0]
    for i in range(2, n):
        reversal = False
        if up_trend:
            p = psar[i - 1] + af * (up_high - psar[i - 1])
            if lo[i] < p:
                reversal, p, down_low, af = True, up_high, lo[i], step
            else:
                if h[i] > up_high:
                    up_high, af = h[i], min(af + step, max_step)
                if lo[i - 2] < p:
                    p = lo[i - 2]
                elif lo[i - 1] < p:
                    p = lo[i - 1]
        else:
            p = psar[i - 1] - af * (psar[i - 1] - down_low)
            if h[i] > p:
                reversal, p, up_high, af = True, down_low, h[i], step
            else:
                if lo[i] < down_low:
                    down_low, af = lo[i], min(af + step, max_step)
                if h[i - 2] > p:
                    p = h[i - 2]
                elif h[i - 1] > p:
                    p = h[i - 1]
        psar[i] = p
        up_trend = up_trend != reversal
        if up_trend:
            up[i] = p
        else:
            down[i] = p
    return up, down


def signals_at(
    ticker: str, frame: pd.DataFrame, i: int, last_bar_complete: bool = False,
) -> Optional[TechnicalSignals]:
    """TechnicalSignals as of row i of a compute_signal_frame result."""
    row = frame.iloc[i]
    values = {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    if any(values[k] is None for k in ("ema_21", "sma_50", "sma_200", "atr_14", "bb_upper", "bb_lower", "rsi_14", "obv")):
        logger.debug("Missing indicator values for %s", ticker)
        return None

    price = values["close"]
    ema_21, sma_50, sma_200 = values["ema_21"], values["sma_50"], values["sma_200"]

    # Parabolic SAR: psar_up is non-NaN when bullish (price above SAR), psar_down when bearish
    if values["psar_up"] is not None:
        psar_val, sar_bearish = values["psar_up"], False
    elif values["psar_down"] is not None:
        psar_val, sar_bearish = values["psar_down"], True
    else:
        psar_val, sar_bearish = price, False

    ref = frame.iloc[i] if last_bar_complete else frame.iloc[i - 1]
    vol_ratio = float(ref["vol_ratio"])
    avg_turnover = float(ref["turnover"])

    rng = values["swing_high"] - values["swing_low"]
    return TechnicalSignals(
        ticker=ticker,
        ema_21=ema_21,
        sma_50=sma_50,
        sma_200=sma_200,
        price_above_50sma=price > sma_50,
        price_above_200sma=price > sma_200,
        ema_21_above_50sma=ema_21 > sma_50,
        atr_14=values["atr_14"],
        bb_upper=values["bb_upper"],
        bb_middle=values["bb_middle"],
        bb_lower=values["bb_lower"],
        bb_pct_b=values["bb_pct_b"] if values["bb_pct_b"] is not None else 0.5,
        rsi_14=values["rsi_14"],
        parabolic_sar=psar_val,
        sar_is_bearish=sar_bearish,
        ease_of_movement=values["eom"] or 0.0,
        obv=values["obv"],
        volume_ratio=vol_ratio,
        volume_spike=vol_ratio >= 1.5,
        fib_38_2=values["swing_high"] - 0.382 * rng,
        fib_61_8=values["swing_high"] - 0.618 * rng,
        current_price=price,
        current_volume=values["volume"],
        avg_turnover=avg_turnover if math.isfinite(avg_turnover) else 0.0,
    )


def _last(series: pd.Series) -> Optional[float]:
    """Return last non-NaN value from a Series, or None."""
    if series is None:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)
