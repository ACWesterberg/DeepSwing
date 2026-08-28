from __future__ import annotations

import logging
from collections import Counter

from config.settings import settings
from src.agent.event_risk import effective_price
from src.analysis.event_model import EventCandidate

logger = logging.getLogger(__name__)


def screen_event_candidates(candidates: list[EventCandidate]) -> list[EventCandidate]:
    """
    Filter priced contracts down to those worth an LLM review, ranked by the edge
    that survives fees.

    The rejection tally is logged at INFO because it is the actual deliverable of
    a dry run: if 'no edge after fees' never fires, the fee model is wrong, and if
    nothing else fires the screener is not screening.
    """
    kept: list[tuple[float, EventCandidate]] = []
    rejections: Counter[str] = Counter()

    for candidate in candidates:
        reason = _reject_reason(candidate)
        if reason is not None:
            rejections[reason] += 1
            logger.debug(
                "REJECT %s: %s (fair=%.3f ask=%.3f edge=%+.3f oi=%d)",
                candidate.ticker, reason, candidate.fair_prob,
                candidate.market_prob, candidate.edge, candidate.contract.open_interest,
            )
            continue
        kept.append((_net_edge(candidate), candidate))

    kept.sort(key=lambda pair: pair[0], reverse=True)
    top = [c for _, c in kept[: settings.max_event_candidates_per_scan]]

    logger.info(
        "Event screener: %d/%d passed%s",
        len(top), len(candidates),
        f" | rejected: {dict(rejections)}" if rejections else "",
    )
    return top


def _net_edge(candidate: EventCandidate) -> float:
    """Edge remaining after the entry fee — the only edge that can be earned."""
    return candidate.fair_prob - effective_price(candidate.market_prob)


def _reject_reason(candidate: EventCandidate) -> str | None:
    contract = candidate.contract

    if not 0.0 < candidate.market_prob < 1.0:
        return "no tradeable ask"
    if contract.yes_bid < settings.min_event_bid:
        return "no bid — one-sided book"
    if contract.open_interest < settings.min_event_open_interest:
        return "thin open interest"
    if contract.spread > settings.max_event_spread:
        return "spread too wide"
    if candidate.edge > settings.max_plausible_edge:
        logger.warning(
            "IMPLAUSIBLE EDGE %s: fair %.3f vs ask %.3f (%+.3f) — forecast %.1fF, "
            "%.2f days lead. Treating as a model fault, not an opportunity.",
            candidate.ticker, candidate.fair_prob, candidate.market_prob,
            candidate.edge, candidate.forecast_high, candidate.lead_days,
        )
        return "edge implausibly large"
    if candidate.edge < settings.min_event_edge:
        return "edge below floor"
    if _net_edge(candidate) <= 0:
        return "no edge after fees"
    return None
