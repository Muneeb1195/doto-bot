"""check_deploy_drift.py — detect drift between the deployed copy and repo HEAD.

The home-server has no .git and is updated by scp, so it can silently drift
from main. This checker runs ON the server (e.g. a daily systemd timer) where
both copies are reachable: it fetches the repo default-branch tarball from
codeload and diffs sha256 hashes of every deploy-managed file against the
local copy. Exits 0 when in sync, 1 on drift (changed/missing/extra files),
2 on operational error.

Deploy-managed = bot/*.py, scripts/*.py, tools/*.py, dashboard/*.py,
dashboard/templates/*.html, services/*.py, pyproject.toml, requirements.txt.
config/settings.ini is deliberately EXCLUDED: it is the live source of truth
on the server (download_models applies CI params) and drifts by design.

Why not in CI: GitHub-hosted runners cannot reach the home-server. CI instead
validates this checker (--self-test job in ci.yml + tests/test_deploy_drift.py);
the detection itself must run where both the local copy and the repo exist.

Usage:
    python scripts/check_deploy_drift.py                     # vs Muneeb1195/doto-bot @ main
    python scripts/check_deploy_drift.py --self-test         # CI: prove the logic works
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "Muneeb1195/doto-bot"
DEFAULT_REF = "main"
UA = "doto-mt5-bot-deploy-drift"

# (relative dir parts, glob) pairs — every file the server is expected to
# mirror from the repo. settings.ini is intentionally absent (live drift).
DEPLOY_PATTERNS: list[tuple[tuple[str, ...], str]] = [
    (("bot",), "*.py"),
    (("scripts",), "*.py"),
    (("tools",), "*.py"),
    (("dashboard",), "*.py"),
    (("dashboard", "templates"), "*.html"),
    (("services",), "*.py"),
]
ROOT_FILES = ["pyproject.toml", "requirements.txt"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_files(root: Path) -> dict[str, str]:
    """relpath -> sha256 for every deploy-managed file under root."""
    out: dict[str, str] = {}
    for parts, pat in DEPLOY_PATTERNS:
        for p in root.joinpath(*parts).glob(pat):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = _sha256(p.read_bytes())
    for name in ROOT_FILES:
        p = root / name
        if p.is_file():
            out[name] = _sha256(p.read_bytes())
    return out


def diff_hashes(local: dict[str, str], repo: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """(changed, missing, extra) — sorted relpaths. `missing` = in repo, not local."""
    changed = sorted(k for k in local.keys() & repo.keys() if local[k] != repo[k])
    missing = sorted(repo.keys() - local.keys())
    extra = sorted(local.keys() - repo.keys())
    return changed, missing, extra


def extract_tarball_hashes(data: bytes) -> dict[str, str]:
    """sha256 of the deploy-managed files inside a codeload tarball.

    Codeload tarballs extract to a single top-level dir (e.g. doto-bot-main/),
    which is stripped so keys match hash_files().
    """
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue  # top-level dir entries / root files without a dir
            rel = Path(*parts[1:]).as_posix()
            # Only keep deploy-managed paths (mirrors hash_files scope).
            if rel not in ROOT_FILES and not any(
                rel.startswith("/".join(d) + "/") and rel.endswith(pat.lstrip("*"))
                for d, pat in DEPLOY_PATTERNS
            ):
                continue
            f = tf.extractfile(member)
            if f is not None:
                out[rel] = _sha256(f.read())
    return out


def fetch_tarball(repo: str, ref: str, token: str | None) -> bytes:
    """Download the repo tarball for `ref` (codeload, no API rate limits)."""
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def run_self_test() -> int:
    """Prove hash/diff/tarball logic on fixtures, with no network. For CI."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "bot").mkdir()
        (root / "scripts").mkdir()
        (root / "config").mkdir()
        (root / "bot" / "main.py").write_text("print('v1')\n")
        (root / "scripts" / "x.py").write_text("X = 1\n")
        (root / "pyproject.toml").write_text("[tool]\n")
        (root / "config" / "settings.ini").write_text("ignored = true\n")

        local = hash_files(root)
        assert "bot/main.py" in local and "scripts/x.py" in local
        assert "pyproject.toml" in local
        assert "config/settings.ini" not in local, "settings.ini must be excluded"

        # In sync: tarball of the same tree must produce an identical map.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in root.rglob("*"):
                if p.is_file():
                    tf.add(p, arcname=f"doto-bot-main/{p.relative_to(root)}")
        repo = extract_tarball_hashes(buf.getvalue())
        changed, missing, extra = diff_hashes(local, repo)
        assert not changed and not missing and not extra, (changed, missing, extra)

        # Drift detection: mutate a tracked file, drop one, add a stray.
        (root / "bot" / "main.py").write_text("print('v2')\n")
        (root / "scripts" / "x.py").unlink()
        (root / "bot" / "stray.py").write_text("S = 1\n")
        local2 = hash_files(root)
        changed2, missing2, extra2 = diff_hashes(local2, repo)
        assert changed2 == ["bot/main.py"], changed2
        assert missing2 == ["scripts/x.py"], missing2
        assert extra2 == ["bot/stray.py"], extra2

        # Tarball extraction must ignore the non-managed stray file too.
        repo2 = extract_tarball_hashes(buf.getvalue())
        assert "bot/stray.py" not in repo2
    print("SELF-TEST PASS: hash, diff, and tarball extraction detect all drift cases")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("DOTO_REPO", DEFAULT_REPO),
                    help=f"owner/name (default {DEFAULT_REPO})")
    ap.add_argument("--ref", default=DEFAULT_REF, help=f"branch/ref (default {DEFAULT_REF})")
    ap.add_argument("--root", default=".", help="deployed copy root (default cwd)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline fixture self-test (CI) and exit")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"[FAIL] --root {root} is not a directory")
        return 2

    token = os.environ.get("GITHUB_TOKEN")
    try:
        local = hash_files(root)
        remote = extract_tarball_hashes(fetch_tarball(args.repo, args.ref, token))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, tarfile.TarError) as e:
        print(f"[ERROR] could not fetch/read repo tarball: {e}")
        return 2

    changed, missing, extra = diff_hashes(local, remote)
    if not (changed or missing or extra):
        print(f"[OK] deployed copy in sync with {args.repo}@{args.ref} "
              f"({len(local)} files)")
        return 0

    print(f"[DRIFT] {args.repo}@{args.ref} differs from the deployed copy:")
    for f in changed:
        print(f"  changed  {f}")
    for f in missing:
        print(f"  missing  {f}")
    for f in extra:
        print(f"  extra    {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
