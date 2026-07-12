from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config.settings import settings
from src.data.options_chain import CONTRACT_MULTIPLIER, OptionContract

logger = logging.getLogger(__name__)


@dataclass
class OptionRiskValidation:
    approved: bool
    contracts: int
    premium_total_sek: float   # total cost at risk (excl. commission)
    reward_risk: float         # profit_target_pct / max_loss_pct
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "contracts": self.contracts,
            "premium_total_sek": self.premium_total_sek,
            "reward_risk": self.reward_risk,
            "rejection_reason": self.rejection_reason,
        }


def validate_option_trade(
    contract: OptionContract,
    premium_sek_per_share: float,
    profit_target_pct: float,
    max_loss_pct: float,
    portfolio_equity: float,
    open_underlyings: list[str],
    is_drawdown_mode: bool = False,
) -> OptionRiskValidation:
    """Validate a proposed long-option entry. The premium paid is the entire risk,
    so sizing caps premium at options_max_premium_pct of equity (hard cap for a
    single unaffordable contract at options_hard_cap_premium_pct)."""

    def _reject(reason: str, rr: float = 0.0) -> OptionRiskValidation:
        return OptionRiskValidation(False, 0, 0.0, rr, reason)

    if premium_sek_per_share <= 0:
        return _reject("Premium quote is zero/invalid")

    reward_risk = profit_target_pct / max_loss_pct if max_loss_pct > 0 else 0.0
    if reward_risk < settings.min_rrr:
        return _reject(
            f"Reward/risk {reward_risk:.2f} (target {profit_target_pct:.0%} / stop {max_loss_pct:.0%}) "
            f"below minimum {settings.min_rrr}", reward_risk,
        )

    if contract.underlying in open_underlyings:
        return _reject(f"Option position already open on {contract.underlying}", reward_risk)

    # Liquidity re-check at decision time — quotes may have moved since the shortlist
    if contract.spread_pct > settings.options_max_spread_pct:
        return _reject(
            f"Spread {contract.spread_pct:.1%} above max {settings.options_max_spread_pct:.0%}", reward_risk,
        )
    if contract.open_interest < settings.options_min_open_interest:
        return _reject(
            f"Open interest {contract.open_interest} below minimum {settings.options_min_open_interest}",
            reward_risk,
        )

    budget = portfolio_equity * settings.options_max_premium_pct
    hard_cap = portfolio_equity * settings.options_hard_cap_premium_pct
    if is_drawdown_mode:
        budget *= 0.5
        hard_cap *= 0.5
        logger.warning("Drawdown mode active — option premium budget halved")

    per_contract = premium_sek_per_share * CONTRACT_MULTIPLIER + settings.options_commission_per_contract_sek
    contracts = math.floor(budget / per_contract)
    if contracts == 0 and per_contract <= hard_cap:
        contracts = 1
    if contracts <= 0:
        return _reject(
            f"One contract costs {per_contract:.0f} SEK — above hard cap "
            f"{hard_cap:.0f} SEK ({settings.options_hard_cap_premium_pct:.0%} of equity)",
            reward_risk,
        )

    premium_total = premium_sek_per_share * CONTRACT_MULTIPLIER * contracts
    return OptionRiskValidation(
        approved=True,
        contracts=contracts,
        premium_total_sek=round(premium_total, 2),
        reward_risk=round(reward_risk, 2),
    )
