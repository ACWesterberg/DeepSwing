from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import settings
from src.agent.candidates import approve_forward_evidence, promote_candidate, register_candidate
from src.agent.compiled_program import program_fingerprint


class SavedProgram:
    def __init__(self, payload):
        self.payload = payload

    def save(self, path):
        Path(path).write_text(json.dumps(self.payload))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: tmp_path))
    candidate = tmp_path / "source.json"
    candidate.write_text(json.dumps({"instructions": "candidate"}))
    return tmp_path, candidate


def test_registration_is_inactive_and_immutable(registry):
    root, source = registry
    active = root / "gpt_trade_decision.json"
    active.write_text(json.dumps({"instructions": "incumbent"}))
    path = register_candidate(
        "gpt", source, run_id="run-1", incumbent_hash=program_fingerprint(active),
        corpus_hash="corpus-1", gate={"promote": True},
    )
    assert json.loads(active.read_text())["instructions"] == "incumbent"
    assert json.loads((path / "manifest.json").read_text())["status"] == "pending_forward_evaluation"
    assert program_fingerprint(path / "program.json") == program_fingerprint(source)


def test_explicit_promotion_is_separate_and_one_way(registry):
    root, source = registry
    candidate_hash = program_fingerprint(source)
    register_candidate(
        "gpt", source, run_id="run-1", incumbent_hash="baseline",
        corpus_hash="corpus-1", gate={"promote": True},
    )
    with pytest.raises(ValueError, match="approved forward evidence"):
        promote_candidate("gpt", candidate_hash, lambda p: None)
    evidence = {"candidate_hash": candidate_hash, "eligible_for_review": True,
                "non_overlapping_periods": 4, "mean_score_gain": 0.12}
    (root / "candidates/gpt" / candidate_hash / "forward_evidence.json").write_text(json.dumps(evidence))
    approve_forward_evidence(
        "gpt", candidate_hash, evidence=evidence,
        approved_by="alex",
    )
    active = promote_candidate("gpt", candidate_hash, lambda p: json.loads(p.read_text()))
    assert program_fingerprint(active) == candidate_hash
    manifest = json.loads((root / "candidates/gpt" / candidate_hash / "manifest.json").read_text())
    assert manifest["status"] == "active"
    with pytest.raises(ValueError, match="approved forward evidence"):
        promote_candidate("gpt", candidate_hash, lambda p: None)


def test_tampered_candidate_cannot_be_promoted(registry):
    root, source = registry
    candidate_hash = program_fingerprint(source)
    path = register_candidate(
        "gpt", source, run_id="run-1", incumbent_hash="baseline",
        corpus_hash="corpus-1", gate={"promote": True},
    )
    evidence = {"candidate_hash": candidate_hash, "eligible_for_review": True}
    (path / "forward_evidence.json").write_text(json.dumps(evidence))
    approve_forward_evidence(
        "gpt", candidate_hash, evidence=evidence,
        approved_by="alex",
    )
    (path / "program.json").write_text(json.dumps({"instructions": "tampered"}))
    with pytest.raises(ValueError, match="fingerprint changed"):
        promote_candidate("gpt", candidate_hash, lambda p: None)
