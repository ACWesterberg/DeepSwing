from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from config.settings import settings
from src.scheduler import backup as backup_mod


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.dev"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


@pytest.fixture
def compiled_dir(tmp_path, monkeypatch):
    """Point settings.compiled_dir at a tmp dir holding a fake compiled program."""
    cdir = tmp_path / "compiled"
    cdir.mkdir()
    (cdir / "claude_trade_decision.json").write_text('{"instructions": "v1", "demos": []}')
    monkeypatch.setattr(type(settings), "compiled_dir", property(lambda self: cdir))
    return cdir


@pytest.fixture(autouse=True)
def _no_push(monkeypatch):
    # Never attempt a network push in tests; commit locally only
    monkeypatch.setattr(settings, "mipro_backup_push", False)
    yield


class TestBackupGuards:
    def test_disabled_when_repo_dir_unset(self, monkeypatch, compiled_dir):
        monkeypatch.setattr(settings, "mipro_backup_repo_dir", "")
        assert backup_mod.backup_compiled_program("claude") is False

    def test_skips_when_repo_dir_not_a_git_repo(self, tmp_path, monkeypatch, compiled_dir):
        plain = tmp_path / "not_a_repo"
        plain.mkdir()
        monkeypatch.setattr(settings, "mipro_backup_repo_dir", str(plain))
        assert backup_mod.backup_compiled_program("claude") is False

    def test_skips_when_compiled_program_missing(self, tmp_path, monkeypatch, compiled_dir):
        repo = tmp_path / "backups"
        _init_git_repo(repo)
        monkeypatch.setattr(settings, "mipro_backup_repo_dir", str(repo))
        # gpt program was never created
        assert backup_mod.backup_compiled_program("gpt") is False


class TestBackupHappyPath:
    def test_commits_program_history_and_metadata(self, tmp_path, monkeypatch, compiled_dir):
        repo = tmp_path / "backups"
        _init_git_repo(repo)
        monkeypatch.setattr(settings, "mipro_backup_repo_dir", str(repo))

        result = backup_mod.backup_compiled_program("claude", {"win_rate": 55.0, "avg_rrr": 2.3})
        assert result is True

        track_dir = repo / "claude"
        assert (track_dir / "latest.json").exists()

        snapshots = list(track_dir.glob("claude_trade_decision_*.json"))
        snapshots = [p for p in snapshots if not p.name.endswith(".meta.json")]
        assert len(snapshots) == 1
        assert json.loads(snapshots[0].read_text())["instructions"] == "v1"

        meta_files = list(track_dir.glob("*.meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["track"] == "claude"
        assert meta["metrics"]["win_rate"] == 55.0

        # The commit actually landed
        log = subprocess.run(
            ["git", "-C", str(repo), "log", "--oneline"],
            capture_output=True, text=True, check=True,
        )
        assert "claude MIPRO program" in log.stdout

    def test_history_accumulates_across_runs(self, tmp_path, monkeypatch, compiled_dir):
        repo = tmp_path / "backups"
        _init_git_repo(repo)
        monkeypatch.setattr(settings, "mipro_backup_repo_dir", str(repo))

        backup_mod.backup_compiled_program("claude")
        # New optimization overwrites the compiled program with a v2
        (compiled_dir / "claude_trade_decision.json").write_text('{"instructions": "v2", "demos": []}')
        backup_mod.backup_compiled_program("claude")

        track_dir = repo / "claude"
        snapshots = [
            p for p in track_dir.glob("claude_trade_decision_*.json")
            if not p.name.endswith(".meta.json")
        ]
        # Both historical versions preserved
        assert len(snapshots) == 2
        # latest reflects v2
        assert json.loads((track_dir / "latest.json").read_text())["instructions"] == "v2"
