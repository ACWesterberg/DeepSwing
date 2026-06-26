from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

TrackType = Literal["claude", "gpt"]


class HeuristicStore:
    """
    File-backed heuristic store with SQLite mirroring.
    Heuristics are stored as JSON files in heuristics/{track}/ and indexed in DB.
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
        all_heuristics = self._load_all()
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
            scored.append((score, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [h for _, h in scored[:top_k]]

        # Update access counts
        for h in top:
            h["access_count"] = h.get("access_count", 0) + 1
            h["last_accessed"] = datetime.utcnow().isoformat()
            file_path = self._dir / f"{h['id']}.json"
            file_path.write_text(json.dumps(h, indent=2))

        return top

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

    def prune(self, quality_threshold: float = 4.0, access_threshold: int = 2) -> int:
        """Remove low-quality, low-access heuristics. Returns count deleted."""
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                h = json.loads(path.read_text())
                if h.get("quality_score", 5.0) < quality_threshold and h.get("access_count", 0) < access_threshold:
                    path.unlink()
                    removed += 1
                    logger.debug("Pruned heuristic %s", h["id"][:8])
            except Exception:
                pass
        if removed:
            logger.info("Pruned %d heuristics from %s track", removed, self.track)
        return removed

    def promote_core(self, access_threshold: int = 10) -> int:
        """Mark frequently-used heuristics as core rules."""
        promoted = 0
        for path in self._dir.glob("*.json"):
            try:
                h = json.loads(path.read_text())
                if not h.get("is_core") and h.get("access_count", 0) >= access_threshold:
                    h["is_core"] = True
                    path.write_text(json.dumps(h, indent=2))
                    promoted += 1
            except Exception:
                pass
        if promoted:
            logger.info("Promoted %d heuristics to core in %s track", promoted, self.track)
        return promoted

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
