from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config.settings import settings
from src.analysis.event_model import EventCandidate

logger = logging.getLogger(__name__)

# All money here is USD, the currency Kalshi contracts are denominated in. The
# caller converts to SEK at the portfolio boundary, so the probability and fee
# arithmetic below never mixes currencies.


@dataclass
class EventRiskValidation:
    approved: bool
    contracts: int
    cost_usd: float          # contracts * price, before fee
    fee_usd: float
    kelly: float             # raw full-Kelly fraction of equity
    effective_price: float   # ask + per-contract fee
    net_edge: float          # fair_prob - effective_price
    rejection_reason: str = ""

    @property
    def total_usd(self) -> float:
        return self.cost_usd + self.fee_usd

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "contracts": self.contracts,
            "cost_usd": round(self.cost_usd, 2),
            "fee_usd": round(self.fee_usd, 2),
            "total_usd": round(self.total_usd, 2),
            "kelly": round(self.kelly, 4),
            "effective_price": round(self.effective_price, 4),
            "net_edge": round(self.net_edge, 4),
            "rejection_reason": self.rejection_reason,
        }


def _reject(reason: str, **kw) -> EventRiskValidation:
    return EventRiskValidation(
        approved=False, contracts=0, cost_usd=0.0, fee_usd=0.0,
        kelly=kw.get("kelly", 0.0),
        effective_price=kw.get("effective_price", 0.0),
        net_edge=kw.get("net_edge", 0.0),
        rejection_reason=reason,
    )


def kalshi_fee(contracts: int, price: float) -> float:
    """
    Kalshi trading fee in USD: ceil(rate * C * P * (1-P)) rounded up to the cent,
    charged on entry only — there is no settlement fee.

    The parabola peaks at P=0.50, where 100 contracts cost $1.75 — a 3.5% haircut
    on stake. This is the term that decides whether a measured edge is real, so it
    is computed exactly rather than approximated.
    """
    if contracts <= 0:
        return 0.0
    raw_cents = settings.kalshi_fee_rate * contracts * price * (1.0 - price) * 100.0
    # Snap off binary-float dust before the ceiling: 100 contracts at 50c computes
    # to 175.00000000000003 cents, which would otherwise bill as $1.76 not $1.75,
    # and makes the fee asymmetric between P and 1-P.
    return math.ceil(round(raw_cents, 9)) / 100.0


def fee_per_contract(price: float) -> float:
    """Unrounded per-contract fee — the marginal cost used for sizing and screening."""
    return settings.kalshi_fee_rate * price * (1.0 - price)


def effective_price(price: float) -> float:
    """What a contract actually costs: ask plus its fee. A contract still pays 1
    on YES, so a fee-inclusive cost is just a binary bet at a worse price."""
    return price + fee_per_contract(price)


def kelly_fraction(fair_prob: float, price: float) -> float:
    """
    Full-Kelly stake as a fraction of equity for a binary contract.

    For a contract costing P that pays 1: f* = (p - P) / (1 - P). Returns 0 when
    the bet has no edge or the price leaves no room.
    """
    if not 0.0 < price < 1.0:
        return 0.0
    return max(0.0, (fair_prob - price) / (1.0 - price))


def validate_event_trade(
    candidate: EventCandidate,
    equity_usd: float,
    open_positions: list[dict],
) -> EventRiskValidation:
    """
    Size one event contract under fractional Kelly with hard caps.

    open_positions entries carry "contract_ticker", "event_ticker" and "cost_usd"
    so duplicate, per-event and portfolio-wide exposure can be enforced.
    """
    price = candidate.market_prob
    if not 0.0 < price < 1.0:
        return _reject(f"Ask {price:.3f} outside (0, 1) — no tradeable quote")

    if equity_usd <= 0:
        return _reject("Equity is zero")

    if any(p.get("contract_ticker") == candidate.ticker for p in open_positions):
        return _reject(f"Position already open in {candidate.ticker}")

    if len(open_positions) >= settings.max_event_positions:
        return _reject(
            f"Already at max_event_positions ({settings.max_event_positions})"
        )

    eff = effective_price(price)
    net_edge = candidate.fair_prob - eff
    ctx = {"effective_price": eff, "net_edge": net_edge}

    if eff >= 1.0:
        return _reject(f"Fee-inclusive price {eff:.3f} leaves no payoff", **ctx)
    if net_edge <= 0:
        return _reject(
            f"No edge after fees: fair {candidate.fair_prob:.3f} vs "
            f"effective price {eff:.3f}",
            **ctx,
        )

    kelly = kelly_fraction(candidate.fair_prob, eff)
    ctx["kelly"] = kelly
    if kelly <= 0:
        return _reject("Kelly fraction is zero", **ctx)

    # Fractional Kelly, then the position cap — whichever binds first.
    stake_usd = min(
        kelly * settings.kelly_fraction * equity_usd,
        settings.max_event_position_pct * equity_usd,
    )

    # Never take more than a share of what is actually resting in the book;
    # open interest is the only depth proxy the public feed gives us.
    depth_cap = int(settings.event_book_depth_fraction * candidate.contract.open_interest)

    # Cap total exposure across every bucket of the same event.
    family_spent = sum(
        p.get("cost_usd", 0.0)
        for p in open_positions
        if p.get("event_ticker") == candidate.contract.event_ticker
    )
    family_room = settings.max_event_family_pct * equity_usd - family_spent
    if family_room <= 0:
        return _reject(
            f"Event {candidate.contract.event_ticker} already at "
            f"{settings.max_event_family_pct:.0%} exposure cap",
            **ctx,
        )
    stake_usd = min(stake_usd, family_room)

    contracts = min(int(stake_usd // eff), depth_cap)
    if contracts < 1:
        return _reject(
            f"Sized below one contract (stake ${stake_usd:.2f}, "
            f"effective price {eff:.3f}, depth cap {depth_cap})",
            **ctx,
        )

    cost_usd = contracts * price
    fee_usd = kalshi_fee(contracts, price)

    return EventRiskValidation(
        approved=True,
        contracts=contracts,
        cost_usd=round(cost_usd, 4),
        fee_usd=fee_usd,
        kelly=kelly,
        effective_price=eff,
        net_edge=net_edge,
    )
