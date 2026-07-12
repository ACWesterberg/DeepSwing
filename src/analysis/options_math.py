from __future__ import annotations

import math
from typing import Literal

RightType = Literal["call", "put"]

_DAYS_PER_YEAR = 365.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, t_years: float, iv: float, r: float) -> tuple[float, float]:
    denom = iv * math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / denom
    return d1, d1 - denom


def bs_price(spot: float, strike: float, dte: int, iv: float, right: RightType, r: float = 0.04) -> float:
    """Black-Scholes option price per share; intrinsic value at/past expiry."""
    if spot <= 0 or strike <= 0:
        return 0.0
    if dte <= 0 or iv <= 0:
        return intrinsic_value(spot, strike, right)
    t = dte / _DAYS_PER_YEAR
    d1, d2 = _d1_d2(spot, strike, t, iv, r)
    if right == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, dte: int, iv: float, right: RightType, r: float = 0.04) -> float:
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        itm = intrinsic_value(spot, strike, right) > 0
        return (1.0 if itm else 0.0) if right == "call" else (-1.0 if itm else 0.0)
    t = dte / _DAYS_PER_YEAR
    d1, _ = _d1_d2(spot, strike, t, iv, r)
    return _norm_cdf(d1) if right == "call" else _norm_cdf(d1) - 1.0


def bs_theta_per_day(spot: float, strike: float, dte: int, iv: float, right: RightType, r: float = 0.04) -> float:
    """Theta in price-per-share per calendar day (negative for long options)."""
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        return 0.0
    t = dte / _DAYS_PER_YEAR
    d1, d2 = _d1_d2(spot, strike, t, iv, r)
    decay = -(spot * _norm_pdf(d1) * iv) / (2.0 * math.sqrt(t))
    if right == "call":
        annual = decay - r * strike * math.exp(-r * t) * _norm_cdf(d2)
    else:
        annual = decay + r * strike * math.exp(-r * t) * _norm_cdf(-d2)
    return annual / _DAYS_PER_YEAR


def intrinsic_value(spot: float, strike: float, right: RightType) -> float:
    if right == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)
