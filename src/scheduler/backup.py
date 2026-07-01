from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def backup_compiled_program(track: str, metrics: Optional[dict] = None) -> bool:
    """
    Copy a track's freshly compiled MIPRO program into the standalone backups
    repo, keeping a full timestamped history plus a `latest.json`, then commit
    (and optionally push). Best-effort: any failure is logged, never raised, so
    a backup problem can't break the optimization run.

    Returns True if a commit was made.
    """
    repo_dir = settings.mipro_backup_repo_dir.strip()
    if not repo_dir:
        logger.debug("MIPRO backup: MIPRO_BACKUP_REPO_DIR unset — skipping")
        return False

    repo = Path(repo_dir).expanduser()
    if not (repo / ".git").is_dir():
        logger.warning("MIPRO backup: %s is not a git working copy — skipping", repo)
        return False

    src = settings.compiled_dir / f"{track}_trade_decision.json"
    if not src.exists():
        logger.warning("MIPRO backup: no compiled program at %s — skipping", src)
        return False

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    dest_dir = repo / track
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "latest.json")
        shutil.copy2(src, dest_dir / f"{track}_trade_decision_{ts}.json")

        meta = {
            "track": track,
            "backed_up_at": datetime.utcnow().isoformat() + "Z",
            "source": str(src),
            "metrics": metrics or {},
        }
        (dest_dir / f"{track}_trade_decision_{ts}.meta.json").write_text(
            json.dumps(meta, indent=2)
        )
    except Exception as exc:
        logger.error("MIPRO backup: failed to stage files for %s: %s", track, exc)
        return False

    return _git_commit_and_push(repo, track, ts)


def _git_commit_and_push(repo: Path, track: str, ts: str) -> bool:
    message = f"Backup {track} MIPRO program {ts}"
    try:
        _run_git(repo, "add", "-A")

        # Nothing staged (identical artifact) → not an error, just skip the commit
        if _run_git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
            logger.info("MIPRO backup: no changes for %s — nothing to commit", track)
            return False

        _run_git(repo, "commit", "-m", message)
        logger.info("MIPRO backup: committed %s to %s", track, repo)

        if settings.mipro_backup_push:
            try:
                _run_git(repo, "push")
                logger.info("MIPRO backup: pushed %s to remote", track)
            except Exception as exc:
                logger.error(
                    "MIPRO backup: commit saved locally but push failed for %s: %s",
                    track, exc,
                )
        return True
    except Exception as exc:
        logger.error("MIPRO backup: git operation failed for %s: %s", track, exc)
        return False


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )
