#!/usr/bin/env python3
"""Download latest trained models/params from GitHub releases.

Runs on the home-server via systemd timer. Checks for new release tags,
downloads artifacts, updates models + strategy params, and restarts the bot.

Uses the GitHub REST API via stdlib urllib — no `gh` CLI or stored PAT needed.
The repo is public, so anonymous API access works (60 req/hr limit).

Environment:
- GITHUB_REPO = owner/repo (default Muneeb1195/doto-bot)
- GITHUB_TOKEN optional (needed only if the repo is ever made private)
- GITHUB_API optional override for the API base (default https://api.github.com)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.environ.get("GITHUB_REPO", "Muneeb1195/doto-bot")
API = os.environ.get("GITHUB_API", "https://api.github.com").rstrip("/")
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TAG_FILE = BASE_DIR / ".last_release_tag"
LOG_FILE = BASE_DIR / "logs" / "download_models.log"


def _setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _api_request(logger, url: str, accept: str = "application/vnd.github+json"):
    """GET a GitHub API URL. Returns parsed JSON, or None on error."""
    req = urllib.request.Request(url, headers={
        "Accept": accept,
        "User-Agent": "doto-mt5-bot",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.error(f"API {e.code} for {url}: {e.read().decode('utf-8', 'replace')[:200]}")
        raise
    except urllib.error.URLError as e:
        logger.error(f"API request failed for {url}: {e}")
        raise


def _latest_release_tag(logger):
    """Latest release tag name, or None if no releases yet."""
    try:
        rel = _api_request(logger, f"{API}/repos/{REPO}/releases/latest")
        return rel.get("tag_name") or None
    except Exception:
        return None


def _download_asset(logger, tag, asset_name, dest):
    """Save a release asset to dest dir. Returns the Path or None."""
    try:
        rel = _api_request(logger, f"{API}/repos/{REPO}/releases/tags/{tag}")
    except Exception:
        return None
    asset = next((a for a in rel.get("assets", []) if a.get("name") == asset_name), None)
    if not asset:
        logger.warning(f"No asset {asset_name} in release {tag}")
        return None
    logger.info(f"Downloading {asset_name} from {tag}...")
    req = urllib.request.Request(asset["browser_download_url"], headers={
        "User-Agent": "doto-mt5-bot",
        "Accept": "application/octet-stream",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return dest
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        logger.warning(f"Failed to download {asset_name}: {e}")
        return None


def _apply_strategy_params(logger):
    """Apply strategy-params.json to settings.ini via update_symbol_strategy.ini."""
    params_path = BASE_DIR / "strategy-params.json"
    if not params_path.exists():
        logger.warning("strategy-params.json not found — skipping param apply")
        return False
    try:
        params = json.loads(params_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read strategy-params.json: {e}")
        return False

    sys.path.insert(0, str(BASE_DIR / "bot"))
    from auto_optimizer import load_portfolio, update_symbol_strategy, write_settings

    _, settings = load_portfolio()
    any_changed = False
    skipped = []
    for symbol, p in params.items():
        if symbol not in settings.get("symbols", []):
            skipped.append(symbol)
            continue
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
    if skipped:
        logger.warning(f"Skipped params for non-portfolio symbols: {sorted(skipped)}")
    return True


def _restart_bot(logger):
    import platform
    if platform.system() == "Linux":
        r = subprocess.run(["systemctl", "--user", "restart", "doto-bot"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            logger.error(f"Bot restart failed: {r.stderr.strip()}")
            return False
        logger.info("Bot restarted")
        return True
    r1 = subprocess.run(["schtasks", "/End", "/TN", "DotoBot"], capture_output=True, text=True)
    if r1.returncode == 0:
        time.sleep(2)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", "DotoBot"], capture_output=True, text=True)
    if r2.returncode != 0:
        logger.error(f"Bot restart failed: {r2.stderr.strip()}")
        return False
    logger.info("Bot restarted")
    return True


def main():
    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("Model/Params Download Check")
    logger.info("=" * 60)

    tag = _latest_release_tag(logger)
    if not tag:
        logger.info("No releases found — nothing to do")
        return 0

    local = TAG_FILE.read_text().strip() if TAG_FILE.exists() else None
    if local == tag:
        logger.info(f"Already at latest release {tag} — nothing to do")
        return 0

    logger.info(f"New release found: {tag} (local: {local})")

    changed = False
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        tar_path = _download_asset(logger, tag, "models.tar.gz", tmp / "models.tar.gz")
        if tar_path:
            MODELS_DIR.mkdir(exist_ok=True)
            extract_dir = tmp / "extract"
            extract_dir.mkdir()
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_dir)
            # The release tar wraps everything under a "models/" prefix; flatten
            # so model_*.pkl / model_*.calib.npz land directly in MODELS_DIR
            # (the path bot/filters.py::load_ml_models actually reads). A nested
            # models/models/ dir silently keeps the bot on stale models.
            extracted = 0
            for member in sorted(extract_dir.rglob("*")):
                if member.is_file() and member.name.startswith("model_"):
                    shutil.move(str(member), MODELS_DIR / member.name)
                    extracted += 1
            logger.info(f"Extracted {extracted} model files to {MODELS_DIR}")
            changed = True

        params_path = _download_asset(logger, tag, "strategy-params.json", tmp / "strategy-params.json")
        if params_path:
            shutil.move(str(params_path), BASE_DIR / "strategy-params.json")
            logger.info("strategy-params.json downloaded")
            changed = True

    if changed:
        # Apply params before restart so the new settings are live immediately.
        _apply_strategy_params(logger)
        TAG_FILE.write_text(tag)
        logger.info(f"Updated local tag to {tag}")
        _restart_bot(logger)
    else:
        logger.warning("No assets downloaded from the new release — not updating tag")

    logger.info("Download check complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())