import configparser
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.plateau_picker import pick_middle_of_plateau

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"auto_optimizer_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

DATA_YEARS = 3
OPTIMIZER_TIMEOUT = 7200


def load_portfolio():
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")
    symbols = [
        s.strip()
        for s in settings.get("PORTFOLIO", "symbols", fallback="").split(",")
        if s.strip()
    ]
    return symbols, settings


def _script_path(name):
    script = BASE_DIR / "bot" / name
    return str(script)


def train_models(csv_mode=False):
    logging.info(f"{'=' * 60}")
    logging.info("Phase 1: Training all ML models")
    logging.info(f"{'=' * 60}")
    trainer = _script_path("train_model.py")
    cmd = [sys.executable, trainer, "--retrain-all", "--pool", "--years", str(DATA_YEARS)]
    if csv_mode:
        cmd.append("--csv")
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    for line in result.stdout.splitlines():
        logging.info(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines()[-20:]:
            logging.error(f"  {line}")
        logging.warning("ML training had errors — continuing anyway")
    else:
        logging.info("ML training complete.")


def optimize_one_symbol(symbol, mode="weekly", csv_mode=False):
    logging.info(f"\n{'─' * 60}")
    logging.info(f"Phase 2: Optimizing {symbol} ({mode})")
    logging.info(f"{'─' * 60}")
    optimizer = _script_path("optimize_params.py")
    years = 3 if mode == "monthly" else DATA_YEARS
    cmd = [
        sys.executable, optimizer,
        "--symbols", symbol,
        "--full-grid" if mode == "weekly" else "--cpcv",
        "--years", str(years),
    ]
    if csv_mode:
        cmd.append("--csv")
    logging.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=OPTIMIZER_TIMEOUT)
    for line in result.stdout.splitlines():
        logging.info(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines()[-20:]:
            logging.error(f"  {line}")
        return None
    csv_path = LOG_DIR / f"optimize_{symbol.replace('.', '_')}.csv"
    if not csv_path.exists():
        logging.warning(f"CSV not found at {csv_path}")
        return None
    return csv_path


optimize_symbol = optimize_one_symbol


def pick_best_params(csv_path, symbol):
    logging.info(f"  Picking plateau params for {symbol}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"  Failed to read CSV: {e}")
        return None
    rec = pick_middle_of_plateau(df, score_col="wf_score", pct=0.90, min_trades=3)
    if rec is None:
        logging.warning(f"  No valid plateau for {symbol}")
        return None
    logging.info(
        f"  Plateau pick: EMA{int(rec['ema_fast'])}/{int(rec['ema_slow'])} "
        f"SL={rec['sl']:.1f} RR={rec['rr']:.1f} ADX={int(rec['adx'])} "
        f"score={rec['score']:.2f} WF={rec['wf_score']:.1f}"
    )
    return rec


def update_symbol_strategy(symbol, rec, settings):
    section = f"STRATEGY:{symbol}"
    if section not in settings:
        settings[section] = {}

    changes = []
    mappings = [
        ("ema_fast", "ema_fast_period", int),
        ("ema_slow", "ema_slow_period", int),
        ("sl", "atr_sl_multiplier", float),
        ("rr", "risk_reward_ratio", float),
        ("adx", "adx_trend_threshold", int),
        ("score", "scoring_min_entry", float),
    ]
    for csv_key, ini_key, transform in mappings:
        val = rec.get(csv_key)
        if val is None:
            continue
        new_val = transform(val)
        current_raw = settings[section].get(ini_key)
        if current_raw is not None:
            try:
                current = transform(current_raw)
            except (ValueError, TypeError):
                current = None
            if current is not None:
                if isinstance(new_val, float):
                    if abs(current - new_val) < 1e-9 * max(abs(new_val), 0.01):
                        continue
                elif current == new_val:
                    continue
        settings[section][ini_key] = str(new_val)
        diff = f"{current_raw}->{new_val}" if current_raw is not None else f"->{new_val}"
        changes.append(f"  {ini_key}: {diff}")

    if not changes:
        logging.info(f"  {symbol}: params unchanged")
        return False

    logging.info(f"  {symbol}: {len(changes)} changes")
    for c in changes:
        logging.info(c)
    return True


def write_settings(settings):
    tmp = Path(str(CONFIG_DIR / "settings.ini") + ".tmp")
    with open(tmp, "w") as f:
        settings.write(f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(CONFIG_DIR / "settings.ini")
    logging.info("settings.ini written")


def stop_bot():
    logging.info("Stopping bot...")
    import platform
    try:
        if platform.system() == "Linux":
            subprocess.run(["systemctl", "stop", "doto-bot"], check=False, timeout=30)
            subprocess.run(["systemctl", "stop", "doto-dashboard"], check=False, timeout=30)
        else:
            subprocess.run(["schtasks", "/End", "/TN", "DotoBot"], check=False, timeout=30)
            subprocess.run(["schtasks", "/End", "/TN", "DotoDashboard"], check=False, timeout=30)
            time.sleep(3)
        logging.info("Bot stopped")
        return True
    except Exception as e:
        logging.error(f"Bot stop error: {e}")
        return False


def restart_bot():
    logging.info("Restarting bot...")
    import platform
    try:
        if platform.system() == "Linux":
            subprocess.run(["systemctl", "restart", "doto-bot"], check=True, timeout=60)
            subprocess.run(["systemctl", "restart", "doto-dashboard"], check=True, timeout=60)
        else:
            subprocess.run(["schtasks", "/End", "/TN", "DotoBot"], check=False, timeout=30)
            time.sleep(2)
            result = subprocess.run(["schtasks", "/Run", "/TN", "DotoBot"], capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logging.error(f"schtasks /Run returned {result.returncode}: {result.stderr.strip()}")
                return False
        logging.info("Bot restart requested")
        return True
    except Exception as e:
        logging.error(f"Bot restart error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Optimizer for Doto MT5 bot")
    parser.add_argument("--apply", action="store_true", help="Apply best params to settings.ini and restart bot")
    parser.add_argument("--skip-train", action="store_true", help="Skip ML model retraining")
    parser.add_argument("--csv", action="store_true", help="Use pre-exported CSV data (no MT5 terminal, for CI)")
    parser.add_argument(
        "--mode", type=str, default="weekly", choices=["weekly", "monthly", "train-only"],
        help="weekly=full-grid (default), monthly=CPCV, train-only=skip optimization",
    )
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("Auto-Optimizer Started")
    logging.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logging.info(f"Mode: {args.mode.upper()} / {'APPLY' if args.apply else 'DRY-RUN'}")
    if args.csv:
        logging.info("CSV MODE: no MT5 terminal — outputting params to JSON release artifact")
    logging.info("=" * 60)

    symbols, settings = load_portfolio()
    if not symbols:
        logging.error("No portfolio symbols found in settings.ini")
        return
    logging.info(f"Portfolio: {', '.join(symbols)} ({len(symbols)} symbols)")

    if not args.skip_train:
        train_models(csv_mode=args.csv)

    if args.mode == "train-only":
        logging.info("train-only mode — optimization skipped")
        return

    if args.apply and not args.csv:
        stop_bot()

    all_best = {}
    for symbol in symbols:
        csv_path = optimize_one_symbol(symbol, mode=args.mode, csv_mode=args.csv)
        if csv_path is None:
            continue
        rec = pick_best_params(csv_path, symbol)
        if rec is not None:
            all_best[symbol] = rec

    if not all_best:
        logging.warning("No valid optimization results for any symbol")
        return

    logging.info(f"\n{'=' * 60}")
    logging.info(f"Results: {len(all_best)}/{len(symbols)} symbols optimized")
    logging.info(f"{'=' * 60}")
    for sym, rec in all_best.items():
        logging.info(
            f"  {sym}: EMA{int(rec['ema_fast'])}/{int(rec['ema_slow'])} "
            f"SL={rec['sl']:.1f} RR={rec['rr']:.1f} ADX={int(rec['adx'])} "
            f"score={rec['score']:.2f} WF={rec['wf_score']:.1f}"
        )

    if args.csv:
        import json
        params_out = {}
        for sym, rec in all_best.items():
            params_out[sym] = {
                "ema_fast_period": int(rec["ema_fast"]),
                "ema_slow_period": int(rec["ema_slow"]),
                "atr_sl_multiplier": float(rec["sl"]),
                "risk_reward_ratio": float(rec["rr"]),
                "adx_trend_threshold": int(rec["adx"]),
                "scoring_min_entry": float(rec.get("score", 0.6)),
            }
        out_path = BASE_DIR / "strategy-params.json"
        out_path.write_text(json.dumps(params_out, indent=2))
        logging.info(f"Strategy params written to {out_path}")
        return

    if args.apply:
        _, settings = load_portfolio()
        any_changed = False
        for symbol, rec in all_best.items():
            if update_symbol_strategy(symbol, rec, settings):
                any_changed = True
        if any_changed:
            write_settings(settings)
            time.sleep(1)
            restart_bot()
        else:
            logging.info("No settings changes — skipping restart")
    else:
        logging.info("Dry-run — no settings changed. Run with --apply to deploy.")

    logging.info("Auto-Optimizer finished")


if __name__ == "__main__":
    main()
