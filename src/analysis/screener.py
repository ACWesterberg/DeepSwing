from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import settings
from src.analysis.regime import RegimeResult
from src.analysis.technical import TechnicalSignals

logger = logging.getLogger(__name__)


@dataclass
class ScreenerCandidate:
    ticker: str
    market: str
    signals: TechnicalSignals
    regime: RegimeResult

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "market": self.market,
            "price": self.signals.current_price,
            "rsi": self.signals.rsi_14,
            "volume_ratio": self.signals.volume_ratio,
            "regime": self.regime.regime,
            "hurst": self.regime.hurst_exponent,
        }


def screen_candidates(
    candidates: dict[str, tuple[TechnicalSignals, RegimeResult]],
    market: str,
) -> list[ScreenerCandidate]:
    """
    Filter a dict of {ticker: (signals, regime)} down to screener-qualified candidates.
    Returns up to settings.max_candidates_per_session sorted by signal strength.
    """
    passed: list[tuple[float, ScreenerCandidate]] = []

    for ticker, (signals, regime) in candidates.items():
        score = _score_candidate(signals, regime)
        if score is None:
            continue
        passed.append((score, ScreenerCandidate(ticker, market, signals, regime)))

    passed.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in passed[: settings.max_candidates_per_session]]
    logger.info("Screener: %d/%d passed for %s market", len(top), len(candidates), market)
    return top


def _score_candidate(signals: TechnicalSignals, regime: RegimeResult) -> float | None:
    """
    Returns a numeric score (higher = stronger candidate) or None to reject.
    Enforces all mandatory filters before scoring.
    """
    def _reject(reason: str) -> None:
        logger.debug("REJECT %s: %s (rsi=%.1f vol=%.2fx regime=%s)", signals.ticker, reason, signals.rsi_14, signals.volume_ratio, regime.regime)

    # --- Mandatory filters ---
    if not signals.price_above_50sma:
        _reject("below 50 SMA")
        return None
    if not (settings.rsi_min <= signals.rsi_14 <= settings.rsi_max):
        _reject(f"RSI {signals.rsi_14:.1f} outside [{settings.rsi_min},{settings.rsi_max}]")
        return None
    if signals.volume_ratio < settings.volume_spike_multiplier:
        _reject(f"volume {signals.volume_ratio:.2f}x < {settings.volume_spike_multiplier}x")
        return None
    # Every level of the trade is denominated in ATR — the stop sits 1.5xATR
    # below entry and the target min_rrr x that above it — so a low-ATR name
    # can only ever pay a small win, and its stop is tight enough that the
    # 25%-of-equity value cap shrinks the position below full risk anyway.
    atr_pct = signals.atr_14 / signals.current_price if signals.current_price > 0 else 0.0
    if settings.min_atr_pct > 0 and atr_pct < settings.min_atr_pct:
        _reject(f"ATR {atr_pct:.2%} < {settings.min_atr_pct:.2%} — too little range to pay a target")
        return None

    # Regime-specific filter: skip neutral regime
    if regime.regime == "neutral":
        _reject("neutral regime")
        return None

    # Regime-specific setup checks
    if regime.regime == "trending":
        # Want EMA21 above SMA50 (bullish momentum structure)
        if not signals.ema_21_above_50sma:
            _reject("trending but EMA21 below SMA50")
            return None
    elif regime.regime == "mean-reverting":
        # Want price near or below BB lower band (pullback into band)
        if signals.bb_pct_b > 0.35:
            _reject(f"mean-rev but bb_pct_b {signals.bb_pct_b:.2f} > 0.35")
            return None

    # --- Score components ---
    score = 0.0

    # Volume conviction
    score += min(signals.volume_ratio - 1.0, 2.0) * 20  # up to +40

    # RSI quality. Centred above the midpoint: a swing entry wants confirmed
    # strength, and rewarding RSI 52.5 while paying zero by 67.5 scored
    # hardest against exactly the names that were moving.
    rsi_mid = 60.0
    score += max(0, 15 - abs(signals.rsi_14 - rsi_mid) * 0.75)  # up to +15

    # Bullish trend alignment
    if signals.price_above_200sma:
        score += 15
    if not signals.sar_is_bearish:
        score += 10

    # Regime confidence (Hurst distance from 0.5 = stronger trend signal)
    score += abs(regime.hurst_exponent - 0.5) * 40  # up to +20 each side

    # Range. Rank up the names that can actually travel far enough to pay a
    # full-size target inside a swing hold; capped so it can't dominate.
    score += min(atr_pct * 100.0, 6.0) * 5  # up to +30

    return score
