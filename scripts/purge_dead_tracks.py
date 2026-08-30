"""
Delete the inert rows and files left by the removed options and
prediction-market tracks.

The options tracks (claude-opt / gpt-opt) were shut down in 2026-08 and the
Kalshi event tracks (claude_events / gpt_events) with them. Nothing reads
their state, but it still shows up in any direct query of portfolio_state or
decisions — and an event track's binary contracts sit alongside equity trades
with R-multiples an order of magnitude larger, which quietly distorts any
aggregate computed across all rows.

/api/reset cannot do this: it validates its targets against settings.tracks
and rejects anything that isn't a live track.

Usage:
    python3 scripts/purge_dead_tracks.py --dry-run
    python3 scripts/purge_dead_tracks.py
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.db import Decision, PortfolioState, get_session  # noqa: E402

DEAD_TRACKS = ["claude-opt", "gpt-opt", "claude_events", "gpt_events"]


def purge(dry_run: bool) -> dict:
    live = set(settings.tracks)
    targets = [t for t in DEAD_TRACKS if t not in live]
    if len(targets) != len(DEAD_TRACKS):
        # A name reappearing in settings.tracks means it was revived; never
        # delete a live track's state on the strength of a stale constant.
        skipped = sorted(set(DEAD_TRACKS) - set(targets))
        print(f"Refusing to purge live track(s): {skipped}")

    report: dict = {}
    session = get_session()
    try:
        for track in targets:
            entry = {
                "portfolio_state_rows": session.query(PortfolioState)
                .filter(PortfolioState.track == track).count(),
                "decision_rows": session.query(Decision)
                .filter(Decision.track == track).count(),
            }

            heuristic_dir = settings.heuristics_dir / track
            entry["heuristic_files"] = (
                len(list(heuristic_dir.glob("*.json"))) if heuristic_dir.exists() else 0
            )

            compiled = sorted(settings.compiled_dir.glob(f"{track}_*.json"))
            entry["compiled_files"] = len(compiled)

            if not dry_run:
                session.query(PortfolioState).filter(
                    PortfolioState.track == track).delete()
                session.query(Decision).filter(Decision.track == track).delete()
                if heuristic_dir.exists():
                    shutil.rmtree(heuristic_dir)
                for path in compiled:
                    path.unlink()

            report[track] = entry

        if not dry_run:
            session.commit()
    finally:
        session.close()

    # Option programs are named for the program, not the track, so they are
    # not caught by the per-track glob above.
    option_programs = sorted(settings.compiled_dir.glob("*option*.json"))
    if option_programs and not dry_run:
        for path in option_programs:
            path.unlink()
    report["_option_programs"] = len(option_programs)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be removed, change nothing")
    args = parser.parse_args()

    report = purge(args.dry_run)
    verb = "Would delete" if args.dry_run else "Deleted"
    for track, counts in report.items():
        if track == "_option_programs":
            continue
        total = sum(counts.values())
        detail = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(f"{verb} for {track}: {detail or 'nothing'}" if total else f"{track}: clean")
    if report.get("_option_programs"):
        print(f"{verb}: {report['_option_programs']} compiled option program(s)")
    if args.dry_run:
        print("\nDry run — nothing changed. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
