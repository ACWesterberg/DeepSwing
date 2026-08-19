"""
Identity for compiled DSPy programs.

MIPRO writes a program and the engine reloads it immediately, so a compile goes
live the moment it finishes. Without a stable identity stamped on the decisions
it makes, there is no way — ever, retrospectively — to ask whether any compile
improved anything: the book's performance moves for a hundred reasons and the
program is only one of them. Fingerprinting each decision is what turns "we
optimize the prompt weekly" into a claim that can be checked.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# What decisions made without a compiled program are grouped under. Every
# compiled version has to beat this arm to have earned its place.
BASELINE = "baseline"


def program_fingerprint(compiled_path: Path) -> str | None:
    """Short stable hash of a compiled program, or None when it can't be read.

    Hashes the parsed JSON with sorted keys rather than the raw bytes, so
    re-saving a program whose content is unchanged does not read as a new
    version — otherwise every weekly compile would look like a new arm and the
    comparison would never accumulate enough trades to say anything.
    """
    try:
        data = json.loads(Path(compiled_path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not fingerprint %s: %s", compiled_path, exc)
        return None

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]
