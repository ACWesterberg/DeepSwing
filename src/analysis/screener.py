from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

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
    sek_factor: Optional[Callable[[str], Optional[float]]] = None,
) -> list[ScreenerCandidate]:
    """
    Filter a dict of {ticker: (signals, regime)} down to screener-qualified candidates.
    Returns up to settings.max_candidates_per_session sorted by signal strength.

    sek_factor(ticker) is the SEK value of one quote unit, used by the liquidity
    floors; it is only called for tickers that clear the technical filters,
    since a London line's unit costs a lookup. None skips the liquidity floors.
    """
    passed: list[tuple[float, ScreenerCandidate]] = []

    for ticker, (signals, regime) in candidates.items():
        score = _score_candidate(signals, regime)
        if score is not None and sek_factor is not None and not _liquid_enough(signals, sek_factor):
            continue
        if score is None:
            continue
        passed.append((score, ScreenerCandidate(ticker, market, signals, regime)))

    passed.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in passed[: settings.max_candidates_per_session]]
    logger.info("Screener: %d/%d passed for %s market", len(top), len(candidates), market)
    return top


def _liquid_enough(signals: TechnicalSignals, sek_factor: Callable[[str], Optional[float]]) -> bool:
    if settings.min_turnover_sek <= 0 and settings.min_price_sek <= 0:
        return True
    factor = sek_factor(signals.ticker)
    if factor is None or factor <= 0:
        logger.debug("REJECT %s: quote currency unresolved", signals.ticker)
        return False
    price_sek = signals.current_price * factor
    if price_sek < settings.min_price_sek:
        logger.debug("REJECT %s: price %.2f SEK < %.2f", signals.ticker, price_sek, settings.min_price_sek)
        return False
    turnover_sek = signals.avg_turnover * factor
    if turnover_sek < settings.min_turnover_sek:
        logger.debug("REJECT %s: turnover %.1f MSEK/day < %.1f", signals.ticker,
                     turnover_sek / 1e6, settings.min_turnover_sek / 1e6)
        return False
    return True


def _score_candidate(signals: TechnicalSignals, regime: RegimeResult) -> float | None:
    """
    Returns a numeric score (higher = stronger candidate) or None to reject.
    Enforces all mandatory filters before scoring.
    """
    def _reject(reason: str) -> None:
        logger.debug("REJECT %s: %s (rsi=%.1f vol=%.2fx regime=%s)", signals.ticker, reason, signals.rsi_14, signals.volume_ratio, regime.regime)

    # --- Mandatory filters: an established uptrend ---
    if settings.require_above_200sma and not signals.price_above_200sma:
        _reject("below 200 SMA")
        return None
    if not signals.price_above_50sma:
        _reject("below 50 SMA")
        return None
    if not signals.ema_21_above_50sma:
        _reject("EMA21 below SMA50")
        return None

    # --- ...that is resting, not bursting ---
    if not (settings.rsi_min <= signals.rsi_14 <= settings.rsi_max):
        _reject(f"RSI {signals.rsi_14:.1f} outside [{settings.rsi_min},{settings.rsi_max}]")
        return None
    if signals.volume_ratio < settings.volume_spike_multiplier:
        _reject(f"volume {signals.volume_ratio:.2f}x < {settings.volume_spike_multiplier}x")
        return None
    if settings.max_volume_ratio > 0 and signals.volume_ratio > settings.max_volume_ratio:
        _reject(f"volume {signals.volume_ratio:.2f}x > {settings.max_volume_ratio}x — spike day")
        return None
    atr_pct = signals.atr_14 / signals.current_price if signals.current_price > 0 else 0.0
    if settings.min_atr_pct > 0 and atr_pct < settings.min_atr_pct:
        _reject(f"ATR {atr_pct:.2%} < {settings.min_atr_pct:.2%}")
        return None
    if settings.max_atr_pct > 0 and atr_pct > settings.max_atr_pct:
        _reject(f"ATR {atr_pct:.2%} > {settings.max_atr_pct:.2%} — gaps through any stop")
        return None

    if settings.regime_gate:
        if regime.regime == "neutral":
            _reject("neutral regime")
            return None
        if regime.regime == "mean-reverting" and signals.bb_pct_b > 0.35:
            _reject(f"mean-rev but bb_pct_b {signals.bb_pct_b:.2f} > 0.35")
            return None

    # --- Score components ---
    score = 0.0

    # Pullback depth: best just above the rising EMA21, fading as price
    # stretches away from it (measured in ATRs so it is scale-free).
    if signals.atr_14 > 0:
        stretch = (signals.current_price - signals.ema_21) / signals.atr_14
        score += max(0.0, 25 - abs(stretch - 0.5) * 10)  # up to +25

    # RSI at rest near the midpoint rather than overbought.
    score += max(0.0, 15 - abs(signals.rsi_14 - 50.0) * 1.0)  # up to +15

    # Trend strength: how far the 50 sits above the 200.
    if signals.sma_200 > 0:
        score += min(max(signals.sma_50 / signals.sma_200 - 1.0, 0.0) * 100, 20.0)  # up to +20

    if not signals.sar_is_bearish:
        score += 5

    return score
