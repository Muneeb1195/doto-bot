#!/usr/bin/env python3
"""Dispatch GitHub workflows and/or download trained models/params.

Runs on the home-server via systemd timer. Two modes:

- --dispatch  (used by doto-orchestrate timer): triggers train.yml, waits for
  it, then triggers optimize.yml (mode=monthly), waits for it, then downloads
  the freshly published models + params, applies them, and restarts the bot.
- --fetch-only / default: pure download check. Pulls models.tar.gz from the
  latest train-* release and strategy-params.json from the latest optimize-*
  release (tracked independently), applies params, restarts the bot if
  anything changed.

Pure-fetch uses the GitHub REST API via stdlib urllib (no gh needed, works on
public repos). --dispatch requires the gh CLI (github-cli) authenticated via
GITHUB_TOKEN.

Environment:
- GITHUB_REPO = owner/repo (default Muneeb1195/doto-bot)
- GITHUB_TOKEN optional for REST fetch; REQUIRED for --dispatch (gh uses it)
- GITHUB_API optional override for the API base (default https://api.github.com)
- GH_BINARY optional override for the gh CLI path (default 'gh')
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
TRAIN_TAG_FILE = BASE_DIR / ".last_train_tag"
OPT_TAG_FILE = BASE_DIR / ".last_optimize_tag"
STREAKS_FILE = BASE_DIR / ".symbol_streaks.json"
LOG_FILE = BASE_DIR / "logs" / "download_models.log"
GH = os.environ.get("GH_BINARY", "gh")

TRAIN_PREFIX = "train-"
OPT_PREFIX = "optimize-"
MODEL_ASSET = "models.tar.gz"
PARAMS_ASSET = "strategy-params.json"
FAILED_PARAMS_ASSET = "failed-params.json"

# Hybrid gate-failure policy: a symbol whose plateau pick fails the DSR/PBO
# gate in the CI publish job lands in failed-params.json. The box re-applies
# its best params with a tightened entry on the FIRST consecutive failure, and
# pauses new entries entirely on the SECOND+ (existing positions still exit).
FAILURE_PAUSE_STRIKE = 2
ENTRY_TIGHTEN_DELTA = 0.15
ENTRY_TIGHTEN_CAP = 0.90

POLL_SECONDS = 30
# How long to wait for each dispatched run before giving up.
RUN_TIMEOUTS = {
    "train.yml": 45 * 60,
    "optimize.yml": 8 * 3600,
}


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


def _latest_tag_for_prefix(logger, prefix):
    """Most recent release tag starting with `prefix`, or None."""
    try:
        rels = _api_request(logger, f"{API}/repos/{REPO}/releases?per_page=30")
        for rel in rels:
            tag = rel.get("tag_name") or ""
            if tag.startswith(prefix):
                return tag
    except Exception:
        pass
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


def _load_streaks():
    """Per-symbol consecutive gate-failure count. Returns dict {sym: int}."""
    if not STREAKS_FILE.exists():
        return {}
    try:
        data = json.loads(STREAKS_FILE.read_text())
        return {str(k): int(v) for k, v in data.items() if int(v) > 0}
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return {}


def _save_streaks(streaks):
    try:
        STREAKS_FILE.write_text(json.dumps(streaks, indent=2))
    except OSError as e:
        logging.getLogger(__name__).warning(f"Failed to persist streak file: {e}")


def _apply_passed_params(logger, settings, params, portfolio):
    """Apply a gate-PASSED symbol's params normally: reset its failure streak
    and force trading_enabled=true (a fresh pass clears any pause)."""
    from auto_optimizer import set_trading_enabled, update_symbol_strategy

    changed = False
    skipped = []
    for symbol, p in params.items():
        if portfolio and symbol not in portfolio:
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
            changed = True
        if set_trading_enabled(symbol, True, settings):
            changed = True
        logger.info(f"  {symbol}: PASS — params applied, trading enabled")
    if skipped:
        logger.warning(f"Skipped params for non-portfolio symbols: {sorted(skipped)}")
    return changed


def _apply_failed_params(logger, settings, failed, streaks):
    """Apply the hybrid policy for gate-failed symbols.

    1st consecutive failure: re-apply best params with a TIGHTENED entry
    (scoring_min_entry + DIFF, capped) — symbol keeps trading.
    2nd+ consecutive failure: pause new entries (trading_enabled=false);
    existing positions still exit. Params not touched on the pause so the
    symbol keeps whatever the last applied (or tightened) config was.
    """
    from auto_optimizer import set_trading_enabled, update_symbol_strategy

    changed = False
    for symbol, p in failed.items():
        streak = streaks.get(symbol, 0) + 1
        streaks[symbol] = streak
        changed = True  # streak state always bumps on a failed release

        if streak >= FAILURE_PAUSE_STRIKE:
            if set_trading_enabled(symbol, False, settings):
                changed = True
            logger.warning(
                f"  {symbol}: FAILED x{streak} — new entries PAUSED "
                f"(existing positions still exit)"
            )
            continue

        rec = {
            "ema_fast": p.get("ema_fast_period"),
            "ema_slow": p.get("ema_slow_period"),
            "sl": p.get("atr_sl_multiplier"),
            "rr": p.get("risk_reward_ratio"),
            "adx": p.get("adx_trend_threshold"),
            "score": p.get("scoring_min_entry"),
        }
        try:
            base_score = float(rec["score"]) if rec["score"] is not None else 0.60
        except (TypeError, ValueError):
            base_score = 0.60
        rec["score"] = min(round(base_score + ENTRY_TIGHTEN_DELTA, 2),
                           ENTRY_TIGHTEN_CAP)
        if update_symbol_strategy(symbol, rec, settings):
            changed = True
        if set_trading_enabled(symbol, True, settings):
            changed = True
        logger.warning(
            f"  {symbol}: FAILED x{streak} — applying params with TIGHTENED "
            f"entry (min_score={rec['score']:.2f}), still trading"
        )
    return changed


def _apply_strategy_params(logger):
    """Apply strategy-params.json (passed) + failed-params.json (hybrid policy)
    to settings.ini. Returns True once run (even if nothing changed)."""
    params_path = BASE_DIR / "strategy-params.json"
    failed_path = BASE_DIR / "failed-params.json"
    params = {}
    failed = {}
    if params_path.exists():
        try:
            params = json.loads(params_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read strategy-params.json: {e}")
    if failed_path.exists():
        try:
            failed = json.loads(failed_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to read failed-params.json: {e}")
    if not params and not failed:
        logger.warning("No strategy-params.json / failed-params.json found — "
                       "skipping param apply")
        return False

    sys.path.insert(0, str(BASE_DIR / "bot"))
    from auto_optimizer import load_portfolio, write_settings

    # load_portfolio() returns (symbols, settings). `settings` is a
    # ConfigParser, whose .get() signature is get(section, option).
    portfolio, settings = load_portfolio()
    streaks = _load_streaks()
    orig_streaks = dict(streaks)

    any_changed = False
    if params and _apply_passed_params(logger, settings, params, portfolio):
        any_changed = True
    if failed and _apply_failed_params(logger, settings, failed, streaks):
        any_changed = True

    if streaks != orig_streaks:
        _save_streaks(streaks)

    if any_changed:
        write_settings(settings)
        logger.info("Strategy params/policy applied to settings.ini")
    else:
        logger.info("No strategy param changes needed")
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


def _extract_models(logger, tar_path):
    MODELS_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        extract_dir = Path(tmp) / "extract"
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
        return extracted > 0


def _gh(logger, *args, check=True):
    """Run gh with the token env inherited. Returns (returncode, stdout)."""
    env = dict(os.environ)
    if os.environ.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    r = subprocess.run([GH, *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        logger.error(f"gh {' '.join(args)} failed: {r.stderr.strip()}")
        if check:
            raise SystemExit(1)
    return r.returncode, r.stdout.strip()


def _dispatch_run(logger, workflow, fields=None):
    """Dispatch workflow, wait for completion. Returns True on success."""
    logger.info(f"Checking for in-flight {workflow} runs...")
    code, out = _gh(logger, "run", "list", "--workflow", workflow, "--repo", REPO,
                    "--limit", "5", "--json", "databaseId,status,workflowName")
    if code == 0 and out:
        try:
            in_flight = [r["databaseId"] for r in json.loads(out)
                         if r.get("status") in ("queued", "in_progress")]
            if in_flight:
                logger.warning(f"Duplicate run guard: {workflow} already in flight "
                               f"(runs {in_flight}) — aborting dispatch")
                return False
        except (json.JSONDecodeError, KeyError):
            pass

    pre = None
    code, out = _gh(logger, "run", "list", "--workflow", workflow, "--repo", REPO,
                    "--limit", "1", "--json", "databaseId")
    if code == 0 and out:
        try:
            pre = json.loads(out)[0]["databaseId"]
        except (json.JSONDecodeError, IndexError, KeyError):
            pre = None

    cmd = ["workflow", "run", workflow, "--repo", REPO, "--ref", "main"]
    for f in (fields or []):
        cmd += ["-f", f]
    _gh(logger, *cmd)
    logger.info(f"Dispatched {workflow}")

    run_id = None
    deadline = time.time() + 5 * 60
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        code, out = _gh(logger, "run", "list", "--workflow", workflow, "--repo", REPO,
                        "--limit", "3", "--json", "databaseId")
        if code == 0 and out:
            try:
                ids = [r["databaseId"] for r in json.loads(out)]
                if pre is None or (ids and ids[0] != pre):
                    run_id = ids[0]
                    break
            except (json.JSONDecodeError, IndexError, KeyError):
                pass
        logger.info("Waiting for the dispatched run to appear in the run list...")
    if not run_id:
        logger.error(f"Could not observe a new {workflow} run after dispatch")
        return False

    logger.info(f"Waiting for {workflow} run {run_id} to complete...")
    deadline = time.time() + RUN_TIMEOUTS.get(workflow, 6 * 3600)
    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        code, out = _gh(logger, "run", "view", str(run_id), "--repo", REPO,
                        "--json", "status,conclusion")
        if code != 0 or not out:
            continue
        try:
            st = json.loads(out)
        except json.JSONDecodeError:
            continue
        status, conclusion = st.get("status"), st.get("conclusion")
        if status == "completed":
            if conclusion == "success":
                logger.info(f"{workflow} run {run_id} completed successfully")
                return True
            logger.error(f"{workflow} run {run_id} ended {conclusion}")
            return False
        logger.info(f"{workflow} run {run_id}: status={status}")
    logger.error(f"Timed out waiting for {workflow} run {run_id}")
    return False


def fetch(logger):
    """Pure download flow. Returns 0 on success (even if nothing to do)."""
    logger.info("=" * 60)
    logger.info("Model/Params Download Check")
    logger.info("=" * 60)

    changed = False

    train_tag = _latest_tag_for_prefix(logger, TRAIN_PREFIX)
    if not train_tag:
        logger.info("No train-* release found — nothing to do")
    else:
        local = TRAIN_TAG_FILE.read_text().strip() if TRAIN_TAG_FILE.exists() else None
        if local == train_tag:
            logger.info(f"Models already at latest train release {train_tag}")
        else:
            logger.info(f"New train release: {train_tag} (local: {local})")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                tar_path = _download_asset(logger, train_tag, MODEL_ASSET, tmp / MODEL_ASSET)
                if tar_path and _extract_models(logger, tar_path):
                    TRAIN_TAG_FILE.write_text(train_tag)
                    logger.info(f"Updated train tag to {train_tag}")
                    changed = True
                else:
                    logger.warning("Model download/extract failed — not updating train tag")

    opt_tag = _latest_tag_for_prefix(logger, OPT_PREFIX)
    if not opt_tag:
        logger.info("No optimize-* release found — nothing to do")
    else:
        local = OPT_TAG_FILE.read_text().strip() if OPT_TAG_FILE.exists() else None
        if local == opt_tag:
            logger.info(f"Params already at latest optimize release {opt_tag}")
        else:
            logger.info(f"New optimize release: {opt_tag} (local: {local})")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                params_path = _download_asset(logger, opt_tag, PARAMS_ASSET,
                                              tmp / PARAMS_ASSET)
                if not params_path:
                    # A release must always carry strategy-params.json; treat a
                    # missing one like a failed download (do not advance tag).
                    logger.warning("Params download failed — not updating optimize tag")
                else:
                    shutil.move(str(params_path), BASE_DIR / PARAMS_ASSET)
                    logger.info("strategy-params.json downloaded")
                    failed_path = _download_asset(logger, opt_tag, FAILED_PARAMS_ASSET,
                                                  tmp / FAILED_PARAMS_ASSET)
                    if failed_path:
                        shutil.move(str(failed_path), BASE_DIR / FAILED_PARAMS_ASSET)
                        logger.info("failed-params.json downloaded")
                    else:
                        # A release that has no gate failures simply omits the
                        # file; remove any leftover so a previous failure isn't
                        # replayed from an older release.
                        stale = BASE_DIR / FAILED_PARAMS_ASSET
                        if stale.exists():
                            stale.unlink()
                            logger.info("No failed-params.json in release — cleared previous")
                    OPT_TAG_FILE.write_text(opt_tag)
                    logger.info(f"Updated optimize tag to {opt_tag}")
                    changed = True

    if changed:
        _apply_strategy_params(logger)
        _restart_bot(logger)
    else:
        logger.info("Nothing new to apply")

    logger.info("Download check complete")
    return 0


def dispatch(logger):
    """Export+push data -> train -> optimize -> fetch, via gh.
    Returns 0 on success."""
    logger.info("=" * 60)
    logger.info("Dispatch + Download")
    logger.info("=" * 60)

    if not os.environ.get("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN not set — required for --dispatch (gh auth)")
        return 1

    if not _push_market_data(logger):
        logger.error("Data export/push failed — aborting cycle")
        return 1

    if not _dispatch_run(logger, "train.yml"):
        logger.error("train.yml failed or already in flight — aborting cycle")
        return 1
    if not _dispatch_run(logger, "optimize.yml", fields=["mode=monthly"]):
        logger.error("optimize.yml failed — aborting cycle")
        return 1

    logger.info("Workflows finished; fetching artifacts...")
    return fetch(logger)


def _push_market_data(logger):
    """Run export_mt5_data.py then push_data.py so the fresh M1 data is on
    GitHub before train.yml starts. Best-effort: a failure here aborts the
    cycle so we never train on stale data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import push_data

    logger.info("Step 0/3: exporting market data from MT5...")
    export_script = Path(__file__).resolve().parent / "export_mt5_data.py"
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, str(export_script), "--no-git"],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        logger.error(f"export_mt5_data.py failed: {r.stderr.strip()[-2000:]}")
        return False
    logger.info("Export complete; pushing M1 data to GitHub release...")
    push_logger = logging.getLogger("push_data")
    push_logger.setLevel(logging.INFO)
    return push_data.push(logging.getLogger("push_data")) == 0


def main():
    logger = _setup_logging()
    flags = {a for a in sys.argv[1:] if a.startswith("-")}

    if "--dispatch" in flags:
        return dispatch(logger)
    # Default and --fetch-only both run the pure download flow.
    return fetch(logger)


if __name__ == "__main__":
    sys.exit(main())
