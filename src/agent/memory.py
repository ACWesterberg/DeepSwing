from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]


def _hours_since(iso_ts: Optional[str], now: datetime) -> float:
    """Hours between an ISO timestamp and now; inf when missing/unparseable."""
    if not iso_ts:
        return float("inf")
    try:
        return (now - datetime.fromisoformat(iso_ts)).total_seconds() / 3600.0
    except ValueError:
        return float("inf")


# Words that carry no discriminating meaning between heuristics — dropping them
# stops two rules looking similar merely because both are English sentences.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those is are was were be been
being to of in on at by for with from as its it their there when while during
not no do does than into over under above below out up down
""".split())

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def canonical_tokens(text: str) -> frozenset[str]:
    """Meaning-bearing words of a heuristic, with every number erased.

    Erasing numbers is the whole point. The library's redundancy is one idea
    minted at a dozen thresholds — "volume exceeds 3x ... RSI below 60" and
    "volume exceeds 5x ... RSI below 55" are the same rule with the dial moved,
    and each was written from a single trade's post-mortem. Compared as written
    they look distinct; with the numbers gone they are visibly one rule.
    """
    stripped = _NUMBER_RE.sub(" ", (text or "").lower())
    words = re.findall(r"[a-z%]+", stripped)
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of two heuristics' canonical tokens, 0.0-1.0."""
    ta, tb = canonical_tokens(a), canonical_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def heuristic_similarity(a: dict, b: dict) -> float:
    """How far two heuristics say the same thing, 0.0-1.0.

    Trigger and action are compared separately and the *lower* score wins, so a
    pair counts as one rule only when both the condition and the response match.
    Two rules can share a condition and prescribe opposite responses — "volume
    spike with price not extended: wait for a pullback" against "volume spike
    with price extended: stand aside" — and collapsing those would delete the
    distinction that makes either useful.
    """
    return min(
        similarity(a.get("trigger", ""), b.get("trigger", "")),
        similarity(a.get("action", ""), b.get("action", "")),
    )


# Two bars, because the two decisions differ in cost and reversibility.
#
# Dedupe marks a rule superseded and takes it out of circulation until someone
# intervenes, so it only fires on near-literal restatements — the same rule with
# the dial moved. Measured across the GPT track's library, those score 1.00.
SIMILARITY_THRESHOLD = 0.70
#
# Retrieval diversity is decided afresh every scan and costs only a slot, so it
# can be stricter about what counts as "already said". On the same library,
# rules within one family (volume spike with price unextended, expressed through
# BB%B, RSI or distance from the SMA) score 0.45-0.50 against each other, while
# every cross-family pair scores 0.00. Cutting at 0.45 keeps a family to one
# seat in the prompt without ever conflating two families.
DIVERSITY_THRESHOLD = 0.45

# Outcomes a heuristic needs before it counts as tested. Below this it still
# gets retrieved — it cannot be measured otherwise — but ranks under rules that
# have been. Now that skipped setups are scored too, outcomes accrue from both
# sides of a decision rather than only from trades that opened.
MIN_CORROBORATION = 2
UNCORROBORATED_PENALTY = 1.5


class HeuristicStore:
    """
    File-backed heuristic store — JSON files in heuristics/{track}/, no DB.
    """

    def __init__(self, track: TrackType):
        self.track = track
        self._dir = settings.heuristics_dir / track
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        trigger: str,
        action: str,
        market: str = "both",
        regime: str = "any",
        quality_score: float = 5.0,
        source_trade_id: Optional[int] = None,
    ) -> str:
        heuristic_id = str(uuid.uuid4())
        heuristic = {
            "id": heuristic_id,
            "track": self.track,
            "trigger": trigger,
            "action": action,
            "market": market,
            "regime": regime,
            "quality_score": quality_score,
            "access_count": 0,
            "is_core": False,
            "created": datetime.utcnow().isoformat(),
            "last_accessed": None,
            "source_trade_id": source_trade_id,
        }

        file_path = self._dir / f"{heuristic_id}.json"
        file_path.write_text(json.dumps(heuristic, indent=2))
        logger.info("Saved heuristic %s for %s track (quality=%.1f)", heuristic_id[:8], self.track, quality_score)
        return heuristic_id

    def retrieve(
        self,
        ticker: str,
        regime: str,
        market: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant heuristics for the current context.
        Relevance = quality_score weighted by regime/market match.
        """
        all_heuristics = [h for h in self._load_all() if not h.get("superseded_by")]
        if not all_heuristics:
            return []

        scored: list[tuple[float, dict]] = []
        for h in all_heuristics:
            score = h["quality_score"]
            if h["regime"] == regime:
                score += 3.0
            elif h["regime"] == "any":
                score += 1.0
            if h["market"] == market:
                score += 2.0
            elif h["market"] == "both":
                score += 1.0
            if h.get("is_core"):
                score += 2.0
            # A heuristic is extracted from a single trade, so a brand new one
            # is an untested guess carrying whatever quality ERL assigned it.
            # Rank it under rules that have actually been measured rather than
            # blocking it outright — it still has to be used to be tested.
            if h.get("outcome_count", 0) < MIN_CORROBORATION:
                score -= UNCORROBORATED_PENALTY
            scored.append((score, h))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Fill the slots with distinct ideas, not the same idea restated. Every
        # heuristic is scored on its own merit, so a cluster of near-identical
        # rules — the library is full of one rule minted at a dozen thresholds —
        # scores alike and can take every slot, leaving the model reading five
        # copies of one consideration and nothing else. Skipping a candidate too
        # close to one already chosen costs a little relevance and buys the
        # model something it did not already have.
        top: list[dict] = []
        for _, h in scored:
            if len(top) >= top_k:
                break
            if any(heuristic_similarity(h, c) >= DIVERSITY_THRESHOLD for c in top):
                continue
            top.append(h)

        # Deliberately no back-filling from the rejected near-duplicates when
        # fewer than top_k distinct rules exist. A fifth restatement of a rule
        # already in the prompt tells the model nothing the first one did not,
        # and costs it the attention a genuinely different consideration would
        # have had. Returning three distinct rules beats returning five copies
        # of one, and a short block is the honest signal that the library holds
        # little the model has not already been told.

        # Update access counts — at most once per hour per heuristic. Retrieval
        # happens every 15-min scan × every candidate, so unthrottled counting
        # inflates counts by hundreds per day and makes the prune/promote
        # thresholds meaningless (rich-get-richer entrenchment of early rules).
        now = datetime.utcnow()
        for h in top:
            h["last_accessed"] = now.isoformat()
            if _hours_since(h.get("last_counted"), now) >= 1.0:
                h["access_count"] = h.get("access_count", 0) + 1
                h["last_counted"] = now.isoformat()
            file_path = self._dir / f"{h['id']}.json"
            file_path.write_text(json.dumps(h, indent=2))

        return top

    def record_outcome(self, heuristic_ids: list[str], pnl_pct: float) -> int:
        """Re-score heuristics against the result of a trade that used them.

        One trade is one unit of evidence, split across the heuristics that were
        in front of the model, weighted by the rank they were retrieved at.
        Giving all of them the identical delta measured "was I present when
        things went well" rather than "did I help": with five retrieved every
        time, a rule accrued the same credit as the four beside it no matter how
        marginal it was, and scores drifted toward a common base rate. Rank is
        the only relevance signal available at this point, and `heuristic_ids`
        arrives in retrieval order, so the top-ranked rule carries the most.

        A heuristic used alone still moves by the full pnl-scaled delta, clamped
        to ±1 and bounded to 0–10.
        """
        full_delta = max(-1.0, min(1.0, pnl_pct * 10.0))
        n = len(heuristic_ids)
        # Linear decay by rank, normalised so the weights sum to 1.
        denom = n * (n + 1) / 2
        weights = [(n - i) / denom for i in range(n)]

        updated = 0
        for rank, heuristic_id in enumerate(heuristic_ids):
            delta = full_delta * weights[rank]
            path = self._dir / f"{heuristic_id}.json"
            if not path.exists():
                continue
            try:
                h = json.loads(path.read_text())
                h["quality_score"] = max(0.0, min(10.0, h.get("quality_score", 5.0) + delta))
                h["outcome_count"] = h.get("outcome_count", 0) + 1
                h["cumulative_pnl_pct"] = round(h.get("cumulative_pnl_pct", 0.0) + pnl_pct, 6)
                path.write_text(json.dumps(h, indent=2))
                updated += 1
            except Exception as exc:
                logger.warning("Failed to record outcome on heuristic %s: %s", heuristic_id[:8], exc)
        if updated:
            logger.info(
                "Outcome %+.2f%% split across %d heuristic(s) in %s track "
                "(top rank Δquality %+.2f)",
                pnl_pct * 100, updated, self.track, full_delta * weights[0],
            )
        return updated

    def to_prompt_text(self, heuristics: list[dict]) -> str:
        if not heuristics:
            return "No relevant heuristics yet."
        lines = []
        for i, h in enumerate(heuristics, 1):
            lines.append(
                f"{i}. IF {h['trigger']} → THEN {h['action']} "
                f"[quality={h['quality_score']:.1f}, used={h['access_count']}x]"
            )
        return "\n".join(lines)

    def dedupe(self, threshold: float = SIMILARITY_THRESHOLD) -> int:
        """Retire heuristics that restate a rule the library already holds.

        The store grows by one rule per post-mortem, so the same idea arrives
        again and again at a slightly different threshold. Left alone it crowds
        retrieval, and every copy accrues its own credit from the same trades.

        Within a cluster the best-evidenced survivor is kept — highest quality,
        then most outcomes behind it, then oldest — and the rest are marked
        superseded rather than deleted, so retrieval skips them while the record
        of what the fund once believed stays intact and reversible.
        """
        live = [h for h in self._load_all() if not h.get("superseded_by")]
        # Strongest first, so a cluster's survivor is chosen before its copies.
        live.sort(
            key=lambda h: (
                h.get("quality_score", 5.0),
                h.get("outcome_count", 0),
                h.get("created", ""),
            ),
            reverse=True,
        )

        kept: list[dict] = []
        superseded = 0
        for h in live:
            match = next(
                (k for k in kept if heuristic_similarity(h, k) >= threshold), None
            )
            if match is None:
                kept.append(h)
                continue
            h["superseded_by"] = match["id"]
            (self._dir / f"{h['id']}.json").write_text(json.dumps(h, indent=2))
            superseded += 1
            logger.debug("Heuristic %s superseded by %s", h["id"][:8], match["id"][:8])

        if superseded:
            logger.info(
                "Deduped %d heuristic(s) in %s track — %d distinct rules remain",
                superseded, self.track, len(kept),
            )
        return superseded

    def prune(
        self,
        quality_threshold: float = 4.0,
        access_threshold: int = 2,
        min_age_days: float = 7.0,
        unused_max_age_days: float = 30.0,
    ) -> int:
        """Remove heuristics that have earned no place, past a grace period.

        Two ways to earn removal. Scoring badly while barely being used is the
        original one, on the `min_age_days` clock. The second is never being
        retrieved at all: quality only moves through `record_outcome`, which
        only fires for heuristics that reached a trade, so a rule that is never
        retrieved sits at its initial 5.0 for ever — above the quality
        threshold, and therefore immortal no matter how long it has
        demonstrated nothing.

        Zero access runs on its own, much longer clock. A rule scoped to a rare
        regime can go a week unretrieved without being dead weight, and would be
        worth having when that regime returns; a month of never once placing in
        any scan's top five is a different claim.
        """
        removed = 0
        now = datetime.utcnow()
        for path in self._dir.glob("*.json"):
            try:
                h = json.loads(path.read_text())
                if _hours_since(h.get("created"), now) < min_age_days * 24:
                    continue
                access = h.get("access_count", 0)
                age_days = _hours_since(h.get("created"), now) / 24.0
                unproven = access == 0 and age_days >= unused_max_age_days
                underperforming = (
                    h.get("quality_score", 5.0) < quality_threshold
                    and access < access_threshold
                )
                if unproven or underperforming:
                    path.unlink()
                    removed += 1
                    logger.debug(
                        "Pruned heuristic %s (%s)", h["id"][:8],
                        "never retrieved" if unproven else "low quality",
                    )
            except Exception:
                pass
        if removed:
            logger.info("Pruned %d heuristics from %s track", removed, self.track)
        return removed

    def promote_core(
        self,
        access_threshold: int = 10,
        quality_threshold: float = 6.0,
        demote_below: float = 4.0,
    ) -> tuple[int, int]:
        """Promote proven rules to core; demote core rules that stopped working.

        Access count measures how often a rule was *shown* to the model, not
        whether it helped — it rises with any rule tagged to a common regime.
        Promoting on that alone handed a permanent +2.0 retrieval boost to
        whatever appeared most, which then made it appear more still. Require
        quality as well, and take the flag back when quality decays, or a rule
        that earned core status once keeps it through any amount of subsequent
        evidence against it.
        """
        promoted = demoted = 0
        for path in self._dir.glob("*.json"):
            try:
                h = json.loads(path.read_text())
                quality = h.get("quality_score", 5.0)
                if h.get("is_core"):
                    if quality < demote_below:
                        h["is_core"] = False
                        path.write_text(json.dumps(h, indent=2))
                        demoted += 1
                elif h.get("access_count", 0) >= access_threshold and quality >= quality_threshold:
                    h["is_core"] = True
                    path.write_text(json.dumps(h, indent=2))
                    promoted += 1
            except Exception:
                pass
        if promoted or demoted:
            logger.info(
                "Core rules in %s track: +%d promoted, -%d demoted",
                self.track, promoted, demoted,
            )
        return promoted, demoted

    def all_as_list(self) -> list[dict]:
        return self._load_all()

    def _load_all(self) -> list[dict]:
        heuristics = []
        for path in self._dir.glob("*.json"):
            try:
                heuristics.append(json.loads(path.read_text()))
            except Exception as exc:
                logger.warning("Failed to load heuristic file %s: %s", path, exc)
        return heuristics


_stores: dict[str, HeuristicStore] = {}


def get_store(track: TrackType) -> HeuristicStore:
    if track not in _stores:
        _stores[track] = HeuristicStore(track)
    return _stores[track]
