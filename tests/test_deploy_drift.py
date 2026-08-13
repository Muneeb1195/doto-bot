"""Tests for `scripts/check_deploy_drift` (the deploy-drift checker).

The checker runs ON the home-server (GitHub runners cannot reach the box), so
CI validates its logic offline: hash scope, diff classification, and codeload
tarball extraction. All tests are pure filesystem/tar operations — no network,
no MT5, no deps beyond the stdlib.
"""

import io
import sys
import tarfile
from pathlib import Path

import check_deploy_drift as cdd


def _make_tree(root: Path) -> None:
    """A minimal deploy-shaped tree: managed dirs + settings.ini (excluded)."""
    for d in ("bot", "scripts", "tools", "dashboard", "dashboard/templates",
              "services", "config"):
        (root / d).mkdir(parents=True)
    (root / "bot" / "main.py").write_text("print('live')\n")
    (root / "scripts" / "push_data.py").write_text("P = 1\n")
    (root / "tools" / "parity_check.py").write_text("T = 1\n")
    (root / "dashboard" / "api.py").write_text("D = 1\n")
    (root / "dashboard" / "templates" / "index.html").write_text("<html></html>\n")
    (root / "services" / "svc.py").write_text("S = 1\n")
    (root / "pyproject.toml").write_text("[tool]\n")
    (root / "requirements.txt").write_text("numpy\n")
    (root / "config" / "settings.ini").write_text("live state — must not drift\n")


def _tar_of(root: Path) -> bytes:
    """Tarball mirroring codeload's layout: single top-level dir + repo tree."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in root.rglob("*"):
            if p.is_file():
                tf.add(p, arcname=f"doto-bot-main/{p.relative_to(root)}")
    return buf.getvalue()


class TestHashScope:
    def test_managed_files_included_settings_excluded(self, tmp_path):
        _make_tree(tmp_path)
        h = cdd.hash_files(tmp_path)
        for rel in ("bot/main.py", "scripts/push_data.py", "tools/parity_check.py",
                    "dashboard/api.py", "dashboard/templates/index.html",
                    "services/svc.py", "pyproject.toml", "requirements.txt"):
            assert rel in h, f"managed file missing from hash map: {rel}"
        assert "config/settings.ini" not in h, "settings.ini must be excluded"
        assert len(h) == 8

    def test_unknown_dirs_ignored(self, tmp_path):
        _make_tree(tmp_path)
        (tmp_path / "config" / "credentials.ini").write_text("secret\n")
        h = cdd.hash_files(tmp_path)
        assert "config/credentials.ini" not in h


class TestDiffHashes:
    def test_classifies_changed_missing_extra(self):
        base = {"bot/main.py": "a", "scripts/x.py": "b", "pyproject.toml": "c"}
        local = {"bot/main.py": "A", "scripts/x.py": "b", "bot/stray.py": "d"}
        changed, missing, extra = cdd.diff_hashes(local, base)
        assert changed == ["bot/main.py"]
        assert missing == ["pyproject.toml"]
        assert extra == ["bot/stray.py"]

    def test_in_sync_is_empty(self):
        h = {"bot/main.py": "a", "pyproject.toml": "c"}
        changed, missing, extra = cdd.diff_hashes(dict(h), dict(h))
        assert not changed and not missing and not extra


class TestTarballExtraction:
    def test_roundtrip_in_sync(self, tmp_path):
        _make_tree(tmp_path)
        local = cdd.hash_files(tmp_path)
        repo = cdd.extract_tarball_hashes(_tar_of(tmp_path))
        changed, missing, extra = cdd.diff_hashes(local, repo)
        assert not changed and not missing and not extra

    def test_drift_detected_through_tarball(self, tmp_path):
        _make_tree(tmp_path)
        repo = cdd.extract_tarball_hashes(_tar_of(tmp_path))
        # Live copy mutates: edit a file, drop one, add a stray.
        (tmp_path / "bot" / "main.py").write_text("print('drifted')\n")
        (tmp_path / "scripts" / "push_data.py").unlink()
        (tmp_path / "bot" / "stray.py").write_text("S = 1\n")
        local = cdd.hash_files(tmp_path)
        changed, missing, extra = cdd.diff_hashes(local, repo)
        assert changed == ["bot/main.py"]
        assert missing == ["scripts/push_data.py"]
        assert extra == ["bot/stray.py"]

    def test_ignores_unmanaged_paths_in_tarball(self, tmp_path):
        _make_tree(tmp_path)
        (tmp_path / "bot" / "main.py").write_text("print('live')\n")
        (tmp_path / "config" / "settings.ini").write_text("x\n")
        repo = cdd.extract_tarball_hashes(_tar_of(tmp_path))
        assert "config/settings.ini" not in repo

    def test_root_files_extracted(self, tmp_path):
        _make_tree(tmp_path)
        repo = cdd.extract_tarball_hashes(_tar_of(tmp_path))
        assert "pyproject.toml" in repo and "requirements.txt" in repo


class TestEntrypoint:
    def test_self_test_mode_returns_zero(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["check_deploy_drift.py", "--self-test"])
        assert cdd.main() == 0

    def test_missing_root_returns_two(self, monkeypatch):
        monkeypatch.setattr(sys, "argv",
                            ["check_deploy_drift.py", "--root", "/nonexistent/xyz"])
        assert cdd.main() == 2
