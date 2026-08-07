#!/usr/bin/env python3
"""Download latest trained models/params from GitHub releases.

Runs on the NUC via systemd timer (hourly). Checks for new release tags,
downloads artifacts, updates models + strategy params, and restarts the bot.

Environment:
- REPO = owner/repo (default Muneeb1195/doto-bot)
- GITHUB_TOKEN optional (for private repos or higher rate limits)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = os.environ.get("GITHUB_REPO", "Muneeb1195/doto-bot")
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TAG_FILE = BASE_DIR / ".last_release_tag"
LOG_FILE = BASE_DIR / "logs" / "download_models.log"


def _setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _latest_release_tag(logger):
    """Return the latest release tag name, or None if no releases."""
    env = os.environ.copy()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        env["GH_TOKEN"] = token
    r = _run(["gh", "release", "view", "--repo", REPO, "--json", "-q", ".tagName"], env=env)
    if r.returncode != 0:
        # Fallback: list releases and get latest
        r = _run(
            ["gh", "release", "list", "--repo", REPO, "--limit", "1",
             "--json", "tagName", "-q", ".[0].tagName"],
            env=env,
        )
        if r.returncode != 0 or not r.stdout.strip():
            logger.warning(f"Could not list releases: {r.stderr.strip()}")
            return None
    tag = r.stdout.strip().strip('"')
    return tag if tag else None


def _local_tag():
    if TAG_FILE.exists():
        return TAG_FILE.read_text().strip()
    return None


def _download_artifact(tag, asset_name, dest, logger):
    """Download a release asset to dest. Returns True on success."""
    env = os.environ.copy()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        env["GH_TOKEN"] = token
    logger.info(f"Downloading {asset_name} from {tag}...")
    r = _run(["gh", "release", "download", tag, "--repo", REPO, "--pattern", asset_name, "--dir", str(dest)], env=env)
    if r.returncode != 0:
        logger.warning(f"Failed to download {asset_name}: {r.stderr.strip()}")
        return False
    return True


def _apply_strategy_params(logger):
    """Apply strategy-params.json to settings.ini via update_symbol_strategy."""
    params_path = BASE_DIR / "strategy-params.json"
    if not params_path.exists():
        return
    try:
        params = json.loads(params_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read strategy-params.json: {e}")
        return

    sys.path.insert(0, str(BASE_DIR / "bot"))
    from auto_optimizer import load_portfolio, update_symbol_strategy, write_settings

    _, settings = load_portfolio()
    any_changed = False
    for symbol, p in params.items():
        # Build rec dict in CSV-key format for update_symbol_strategy
        rec = {
            "ema_fast": p.get("ema_fast_period"),
            "ema_slow": p.get("ema_slow_period"),
            "sl": p.get("atr_sl_multiplier"),
            "rr": p.get("risk_reward_ratio"),
            "adx": p.get("adx_trend_threshold"),
            "score": p.get("scoring_min_entry"),
        }
        if update_symbol_strategy(symbol, rec, settings):
            any_changed = True

    if any_changed:
        write_settings(settings)
        logger.info("Strategy params applied to settings.ini")
    else:
        logger.info("No strategy param changes needed")


def _restart_bot(logger):
    import platform
    if platform.system() == "Linux":
        r = _run(["systemctl", "--user", "restart", "doto-bot"])
    else:
        r = _run(["schtasks", "/End", "/TN", "DotoBot"])
        if r.returncode == 0:
            import time
            time.sleep(2)
        r = _run(["schtasks", "/Run", "/TN", "DotoBot"])
    if r.returncode != 0:
        logger.error(f"Bot restart failed: {r.stderr.strip()}")
        return False
    logger.info("Bot restarted")
    return True


def main():
    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("Model/Params Download Check")
    logger.info("=" * 60)

    tag = _latest_release_tag(logger)
    if tag is None:
        logger.info("No releases found — nothing to do")
        return

    local = _local_tag()
    if local == tag:
        logger.info(f"Already at latest release {tag} — nothing to do")
        return

    logger.info(f"New release found: {tag} (local: {local})")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Download models tar.gz
        models_ok = _download_artifact(tag, "models.tar.gz", tmp, logger)
        if models_ok:
            tar_path = tmp / "models.tar.gz"
            if tar_path.exists():
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(MODELS_DIR)
                logger.info(f"Extracted models to {MODELS_DIR}")

        # Download strategy params
        params_ok = _download_artifact(tag, "strategy-params.json", BASE_DIR, logger)

    # Apply strategy params if downloaded
    if params_ok:
        _apply_strategy_params(logger)

    # Update local tag
    TAG_FILE.write_text(tag)
    logger.info(f"Updated local tag to {tag}")

    # Restart bot if anything changed
    if models_ok or params_ok:
        _restart_bot(logger)

    logger.info("Download check complete")


if __name__ == "__main__":
    sys.exit(main())
