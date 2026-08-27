from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Observed daily highs are whole degrees and stations have their own noise, so
# forecast uncertainty never collapses to zero however close to resolution we are.
SIGMA_FLOOR_F = 0.5

# Kalshi strike types seen on the KXHIGH* daily-high series.
_UPPER_OPEN = {"greater", "greater_or_equal"}
_LOWER_OPEN = {"less", "less_or_equal"}


@dataclass
class EventContract:
    """One tradeable Kalshi binary market — a single strike bucket of one event.

    Prices are probability units (0..1), converted from Kalshi's integer cents by
    the client. Strikes are degrees Fahrenheit.
    """
    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    yes_bid: float
    yes_ask: float
    last_price: float
    volume: int
    open_interest: int
    close_time: datetime
    strike_type: str
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)

    @property
    def mid(self) -> float:
        if self.yes_bid <= 0 and self.yes_ask <= 0:
            return self.last_price
        return (self.yes_bid + self.yes_ask) / 2.0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "title": self.title,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "spread": round(self.spread, 4),
            "open_interest": self.open_interest,
            "close_time": self.close_time.isoformat(),
        }


@dataclass
class EventCandidate:
    """A contract whose model probability diverges from its ask."""
    contract: EventContract
    fair_prob: float
    market_prob: float   # the ask — what a taker actually pays
    edge: float          # fair_prob - market_prob
    forecast_high: float
    sigma: float
    lead_days: float

    @property
    def ticker(self) -> str:
        return self.contract.ticker

    def to_dict(self) -> dict:
        return {
            **self.contract.to_dict(),
            "fair_prob": round(self.fair_prob, 4),
            "market_prob": round(self.market_prob, 4),
            "edge": round(self.edge, 4),
            "forecast_high": round(self.forecast_high, 1),
            "sigma": round(self.sigma, 2),
            "lead_days": round(self.lead_days, 2),
        }

    def to_prompt_str(self) -> str:
        """Numeric context for the LLM veto. Deliberately states the model's own
        probability rather than asking the model to produce one."""
        lo, hi = bucket_bounds(self.contract)
        lo_s = "-inf" if lo == -math.inf else f"{lo:.1f}"
        hi_s = "+inf" if hi == math.inf else f"{hi:.1f}"
        return (
            f"Contract: {self.contract.ticker} ({self.contract.title})\n"
            f"Resolves YES if the daily high falls in [{lo_s}, {hi_s}] degF\n"
            f"NWS forecast high: {self.forecast_high:.1f} degF "
            f"(sigma {self.sigma:.2f} degF at {self.lead_days:.2f} days lead)\n"
            f"Model fair probability: {self.fair_prob:.3f}\n"
            f"Market: bid {self.contract.yes_bid:.2f} / ask {self.contract.yes_ask:.2f} "
            f"(spread {self.contract.spread:.3f})\n"
            f"Edge vs ask: {self.edge:+.3f}\n"
            f"Open interest: {self.contract.open_interest}, volume: {self.contract.volume}\n"
            f"Hours to close: {self.lead_days * 24:.1f}"
        )


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast_sigma(lead_days: float) -> float:
    """
    Forecast-error standard deviation in degF as a function of lead time.

    Below one day error grows roughly with sqrt(lead); beyond it, linearly. The
    two branches meet at lead=1, so the curve is continuous. Seeded from published
    NWS MAE — recalibrate from observed error once contracts have settled.
    """
    lead = max(0.0, lead_days)
    day1 = settings.forecast_sigma_day1
    if lead <= 1.0:
        return max(SIGMA_FLOOR_F, day1 * math.sqrt(lead))
    return day1 + settings.forecast_sigma_per_day * (lead - 1.0)


def bucket_bounds(contract: EventContract) -> tuple[float, float]:
    """
    Continuous [lo, hi) bounds for a contract's YES region, in degF.

    Daily highs resolve as whole degrees, so a bucket quoted on integer strikes
    covers half a degree either side — without that continuity correction a
    "82 to 83" bucket would be modelled as an interval of width 1 instead of 2,
    understating its probability by roughly half.
    """
    floor_s, cap_s = contract.floor_strike, contract.cap_strike
    strike_type = (contract.strike_type or "").lower()

    if strike_type in _UPPER_OPEN or (floor_s is not None and cap_s is None):
        if floor_s is None:
            raise ValueError(f"{contract.ticker}: {strike_type} strike without floor_strike")
        return _correct(floor_s, inclusive=strike_type != "greater"), math.inf

    if strike_type in _LOWER_OPEN or (cap_s is not None and floor_s is None):
        if cap_s is None:
            raise ValueError(f"{contract.ticker}: {strike_type} strike without cap_strike")
        return -math.inf, _correct(cap_s, inclusive=strike_type != "less", upper=True)

    if floor_s is None or cap_s is None:
        raise ValueError(f"{contract.ticker}: unusable strike ({strike_type}, {floor_s}, {cap_s})")

    return _correct(floor_s, inclusive=True), _correct(cap_s, inclusive=True, upper=True)


def _correct(strike: float, *, inclusive: bool, upper: bool = False) -> float:
    """Shift an integral strike to the half-degree boundary that bounds it."""
    if strike != math.floor(strike):
        return strike  # already a half-degree boundary — Kalshi encoded it for us
    if upper:
        return strike + 0.5 if inclusive else strike - 0.5
    return strike - 0.5 if inclusive else strike + 0.5


def bucket_probability(lo: float, hi: float, mu: float, sigma: float) -> float:
    """P(lo <= X < hi) for X ~ Normal(mu, sigma)."""
    if sigma <= 0:
        return 1.0 if lo <= mu < hi else 0.0
    upper = 1.0 if hi == math.inf else _phi((hi - mu) / sigma)
    lower = 0.0 if lo == -math.inf else _phi((lo - mu) / sigma)
    return max(0.0, min(1.0, upper - lower))


def fair_probability(
    contract: EventContract,
    forecast_high: float,
    now: Optional[datetime] = None,
) -> tuple[float, float, float]:
    """Fair probability for one contract. Returns (prob, sigma, lead_days)."""
    now = now or datetime.utcnow()
    lead_days = max(0.0, (contract.close_time - now).total_seconds() / 86400.0)
    sigma = forecast_sigma(lead_days)
    lo, hi = bucket_bounds(contract)
    return bucket_probability(lo, hi, forecast_high, sigma), sigma, lead_days


def build_candidates(
    contracts: list[EventContract],
    forecast_by_event: dict[str, float],
    now: Optional[datetime] = None,
    normalize: bool = True,
) -> list[EventCandidate]:
    """
    Price every contract against its event's forecast high.

    Buckets within one event are mutually exclusive and exhaustive, so when the
    full ladder is present their probabilities are renormalised to sum to 1. A
    partial ladder is left alone — rescaling it would inflate every probability.
    """
    now = now or datetime.utcnow()
    by_event: dict[str, list[EventCandidate]] = {}

    for contract in contracts:
        forecast_high = forecast_by_event.get(contract.event_ticker)
        if forecast_high is None:
            logger.debug("No forecast for event %s — skipping", contract.event_ticker)
            continue
        try:
            prob, sigma, lead_days = fair_probability(contract, forecast_high, now)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", contract.ticker, exc)
            continue
        by_event.setdefault(contract.event_ticker, []).append(
            EventCandidate(
                contract=contract,
                fair_prob=prob,
                market_prob=contract.yes_ask,
                edge=prob - contract.yes_ask,
                forecast_high=forecast_high,
                sigma=sigma,
                lead_days=lead_days,
            )
        )

    candidates: list[EventCandidate] = []
    for event_ticker, group in by_event.items():
        if normalize:
            _normalize_group(event_ticker, group)
        candidates.extend(group)
    return candidates


def _normalize_group(event_ticker: str, group: list[EventCandidate]) -> None:
    """Rescale a complete bucket ladder to sum to 1, in place."""
    total = sum(c.fair_prob for c in group)
    if total <= 0:
        return
    # A complete ladder already sums to ~1; anything far off means buckets are
    # missing from the feed, and rescaling would overstate every one of them.
    if not 0.90 <= total <= 1.10:
        logger.debug(
            "Event %s bucket probabilities sum to %.3f — partial ladder, not normalising",
            event_ticker, total,
        )
        return
    for candidate in group:
        candidate.fair_prob /= total
        candidate.edge = candidate.fair_prob - candidate.market_prob
