"""Immutable prompt-candidate registry and explicit activation boundary."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

from config.settings import settings
from src.agent.compiled_program import program_fingerprint, save_compiled_program
from src.agent.evaluation import write_json_atomic


def candidate_dir(track: str, candidate_hash: str) -> Path:
    return settings.compiled_dir / "candidates" / track / candidate_hash


def register_candidate(
    track: str,
    candidate_path: Path,
    *,
    run_id: str,
    incumbent_hash: str,
    corpus_hash: str,
    gate: dict,
) -> Path:
    """Register a historically screened candidate without activating it."""
    candidate_hash = program_fingerprint(candidate_path)
    if not candidate_hash:
        raise ValueError("Cannot register an unreadable candidate")
    root = candidate_dir(track, candidate_hash)
    program_path = root / "program.json"
    manifest_path = root / "manifest.json"
    manifest = {
        "version": 1,
        "track": track,
        "candidate_hash": candidate_hash,
        "incumbent_hash_at_registration": incumbent_hash,
        "historical_run_id": run_id,
        "historical_corpus_hash": corpus_hash,
        "historical_gate": gate,
        "registered_at": datetime.utcnow().isoformat(),
        "status": "pending_forward_evaluation",
        "activated_at": None,
    }
    if root.exists():
        existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        same_identity = existing and all(
            existing.get(key) == manifest[key]
            for key in ("version", "track", "candidate_hash")
        )
        if not same_identity or program_fingerprint(program_path) != candidate_hash:
            raise ValueError("Candidate hash already exists with different content")
        return root
    root.mkdir(parents=True)
    shutil.copy2(candidate_path, program_path)
    if program_fingerprint(program_path) != candidate_hash:
        raise ValueError("Registered candidate does not match its fingerprint")
    write_json_atomic(manifest_path, manifest)
    return root


def approve_forward_evidence(
    track: str,
    candidate_hash: str,
    *,
    evidence: dict,
    approved_by: str,
) -> Path:
    """Record an explicit human approval of a candidate's forward evidence."""
    if not approved_by.strip():
        raise ValueError("Forward approval requires an approver")
    root = candidate_dir(track, candidate_hash)
    program_path = root / "program.json"
    manifest_path = root / "manifest.json"
    if not program_path.exists() or not manifest_path.exists():
        raise ValueError("Candidate is not registered")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "pending_forward_evaluation":
        raise ValueError("Candidate is not pending forward evaluation")
    if program_fingerprint(program_path) != candidate_hash:
        raise ValueError("Candidate artifact fingerprint changed")
    evidence_path = root / "forward_evidence.json"
    stored_evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None
    if not isinstance(evidence, dict) or evidence.get("candidate_hash") != candidate_hash:
        raise ValueError("Forward approval requires evidence for this candidate")
    if evidence != stored_evidence:
        raise ValueError("Forward approval must use the stored evidence report")
    if not evidence.get("eligible_for_review"):
        raise ValueError("Forward evidence has not met the configured review floor")
    manifest["status"] = "forward_approved"
    manifest["forward_approval"] = {
        "approved_at": datetime.utcnow().isoformat(),
        "approved_by": approved_by.strip(),
        "evidence": evidence,
    }
    write_json_atomic(manifest_path, manifest)
    return root


def promote_candidate(track: str, candidate_hash: str, validate: Callable[[Path], None]) -> Path:
    """Explicitly activate a forward-approved candidate.

    Scheduled optimization never calls this function. A separate approval step
    must first bind the stored eligible evidence report to a human approver.
    """
    root = candidate_dir(track, candidate_hash)
    program_path = root / "program.json"
    manifest_path = root / "manifest.json"
    if not program_path.exists() or not manifest_path.exists():
        raise ValueError("Candidate is not registered")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "forward_approved" or not manifest.get("forward_approval"):
        raise ValueError("Candidate does not have approved forward evidence")
    if program_fingerprint(program_path) != candidate_hash:
        raise ValueError("Candidate artifact fingerprint changed")
    active = settings.compiled_dir / f"{track}_trade_decision.json"
    save_compiled_program(_SavedProgram(program_path), active, validate)
    manifest["status"] = "active"
    manifest["activated_at"] = datetime.utcnow().isoformat()
    write_json_atomic(manifest_path, manifest)
    return active


class _SavedProgram:
    """Adapter allowing atomic persistence to copy a validated JSON artifact."""

    def __init__(self, source: Path):
        self.source = source

    def save(self, path: str) -> None:
        shutil.copy2(self.source, path)
