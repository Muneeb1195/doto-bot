#!/usr/bin/env python3
"""Upload exported M1 market-data CSVs to a GitHub release and prune old ones.

Runs on the home-server after `export_mt5_data.py --no-git` as part of the
monthly doto-orchestrate cycle (or manually). Each `data/history/*_M1.csv` is
uploaded as an INDIVIDUAL asset to a freshly created `data-YYYYMMDD-HHMM`
release, so train.yml / optimize.yml shards can `gh release download --pattern
'<SYM>_M1.csv'` exactly the files they need without a git checkout of the ~2GB
M1 history (Git LFS bandwidth/space is a non-issue for this repo).

Prunes all but the NEWEST 2 `data-*` releases (release + tag), since each cycle
re-exports everything anyway — old M1 archives only waste quota.

Environment:
- GITHUB_REPO = owner/repo (default Muneeb1195/doto-bot)
- GITHUB_TOKEN REQUIRED (gh auth)
- GH_BINARY optional override for the gh CLI path (default 'gh')
- DATA_DIR optional override for the data directory (default ../data/history)
- KEEP_RELEASES optional count to retain (default 2)

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = os.environ.get("GITHUB_REPO", "Muneeb1195/doto-bot")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data" / "history"))
LOG_FILE = BASE_DIR / "logs" / "download_models.log"
GH = os.environ.get("GH_BINARY", "gh")
KEEP_RELEASES = int(os.environ.get("KEEP_RELEASES", "2"))
PREFIX = "data-"
ASSET_GLOB = "*_M1.csv"


def _setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _gh(logger, *args, check=True):
    env = dict(os.environ)
    if os.environ.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    r = subprocess.run([GH, *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        logger.error(f"gh {' '.join(args)} failed: {r.stderr.strip()}")
        if check:
            raise SystemExit(1)
    return r.returncode, r.stdout.strip()


def _list_data_releases(logger):
    """Return list of {tag, id} for data-* releases, newest first."""
    code, out = _gh(logger, "release", "list", "--repo", REPO, "--limit", "50",
                    "--json", "tagName,id")
    if code != 0:
        return []
    try:
        return [{"tag": r["tagName"], "id": r["id"]}
                for r in json.loads(out) if r.get("tagName", "").startswith(PREFIX)]
    except (json.JSONDecodeError, KeyError):
        logger.error("Failed to parse gh release list output")
        return []


def push(logger):
    logger.info("=" * 60)
    logger.info("Push M1 data to GitHub release")
    logger.info("=" * 60)

    if not os.environ.get("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN not set — required for data push (gh auth)")
        return 1

    assets = sorted(DATA_DIR.glob(ASSET_GLOB))
    if not assets:
        logger.error(f"No {ASSET_GLOB} files in {DATA_DIR} — nothing to push")
        return 1

    total_mb = sum(a.stat().st_size for a in assets) / (1 << 20)
    logger.info(f"Found {len(assets)} M1 CSVs ({total_mb:.0f} MB) in {DATA_DIR}")

    tag = f"{PREFIX}{datetime.now().strftime('%Y%m%d-%H%M')}"
    logger.info(f"Creating release {tag}...")
    _gh(logger, "release", "create", tag, "--repo", REPO,
        "--title", f"Market Data {tag}", "--notes",
        "M1 market-data CSVs for the ML training / optimization cycle. "
        "Uploaded automatically by scripts/push_data.py after export.")

    uploaded = 0
    for asset in assets:
        _gh(logger, "release", "upload", tag, str(asset), "--repo", REPO,
            "--clobber")
        uploaded += 1
        logger.info(f"  uploaded {asset.name}")
    logger.info(f"Uploaded {uploaded} assets to {tag}")

    releases = _list_data_releases(logger)
    if len(releases) > KEEP_RELEASES:
        stale = releases[KEEP_RELEASES:]
        logger.info(f"Pruning {len(stale)} old data releases (keeping {KEEP_RELEASES})...")
        for rel in stale:
            _gh(logger, "release", "delete", rel["tag"], "--repo", REPO, check=False)
            _gh(logger, "api", "-X", "DELETE",
                f"repos/{REPO}/git/refs/tags/{rel['tag']}", check=False)
            logger.info(f"  deleted {rel['tag']}")
    else:
        logger.info(f"{len(releases)} data releases present — nothing to prune")

    logger.info("Data push complete")
    return 0


def main():
    logger = _setup_logging()
    return push(logger)


if __name__ == "__main__":
    sys.exit(main())
