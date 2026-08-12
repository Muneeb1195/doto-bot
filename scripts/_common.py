"""Shared helpers for scripts/ entry points.

The scripts in this directory are run as standalone programs
(`python scripts/x.py`), which puts `scripts/` on sys.path[0], so they
import this module with a plain `from _common import ...`.

Deduplicated here: `_gh` (gh CLI wrapper with GITHUB_TOKEN -> GH_TOKEN env
inheritance) and `_setup_logging` (file + console logging to a per-script
log file). Previously copy-pasted into download_models.py, push_data.py and
export_mt5_data.py.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path


def _setup_logging(log_file: Path):
    """Configure file + console logging to log_file. Returns a logger."""
    log_file = Path(log_file)
    log_file.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _gh(logger, *args, check=True):
    """Run gh with the token env inherited. Returns (returncode, stdout)."""
    gh = os.environ.get("GH_BINARY", "gh")
    env = dict(os.environ)
    if os.environ.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    r = subprocess.run([gh, *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        logger.error(f"gh {' '.join(args)} failed: {r.stderr.strip()}")
        if check:
            raise SystemExit(1)
    return r.returncode, r.stdout.strip()
