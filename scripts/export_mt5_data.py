#!/usr/bin/env python3
"""Export MT5 terminal bars to CSV files for GitHub Actions training/optimization.

Requires a running MT5 terminal with RPyC bridge. Run on the NUC via systemd
timer every 6 hours. Exports H1 (regular git) + M15 (regular git) + M1 (Git LFS)
for each portfolio symbol.

Gracefully no-ops if MT5 is unavailable (doesn't crash the timer).
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
HISTORY_DIR = BASE_DIR / "data" / "history"
LOG_FILE = BASE_DIR / "logs" / "export_mt5_data.log"

# 8 live-portfolio symbols + 13 asset-class pool symbols. The home-server
# dispatches export_mt5_data.py (no args) each monthly cycle and uploads the
# M1 files to a data-* release; train.yml trains the pools on ALL symbols, so
# the pool symbols must be exported too even though the bot does not trade them.
SYMBOLS = [
    "BTCUSD.raw", "US30.raw", "GBPJPY.raw", "SOLUSD.raw",
    "XRPUSD.raw", "EURUSD.raw", "US500.raw", "XAUUSD.raw",
    "XAGUSD.raw", "XNGUSD.raw", "XPTUSD.raw", "XAU500.raw",
    "US100.raw", "UK100.raw", "JP225.raw",
    "ETHUSD.raw", "DOGUSD.raw", "USDJPY.raw", "GBPUSD.raw",
    "AUDUSD.raw", "USDCAD.raw",
]

# A single copy_rates_range is capped at ~100k bars by the terminal, which
# silently truncates any timeframe whose window exceeds that. At 5.1 years only
# H1 (~45k bars) fits; M15 (~178k) and M1 (~2.7M) must page.
TF_MAP = {
    "H1": {"tf_attr": "TIMEFRAME_H1", "use_paged": False},
    "M15": {"tf_attr": "TIMEFRAME_M15", "use_paged": True},
    "M1": {"tf_attr": "TIMEFRAME_M1", "use_paged": True},
}

EXPORT_YEARS = 5.1


def _setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_settings():
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")
    return settings


def _save_symbol_info(settings, symbol, point, tick_value, volume_step):
    if not settings.has_section("SYMBOL_POINTS"):
        settings.add_section("SYMBOL_POINTS")
    settings.set("SYMBOL_POINTS", symbol, str(point))
    settings.set("SYMBOL_POINTS", symbol + "_tick", str(tick_value))
    settings.set("SYMBOL_POINTS", symbol + "_vstep", str(volume_step))
    tmp = Path(str(CONFIG_DIR / "settings.ini") + ".tmp")
    with open(tmp, "w") as f:
        settings.write(f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(CONFIG_DIR / "settings.ini")


def _export_tf(mt5_mod, symbol, tf_name, tf_config, logger):
    """Export one timeframe for one symbol. Returns (rows, info) or (None, None)."""
    tf = getattr(mt5_mod, tf_config["tf_attr"])

    end = datetime.now()
    start = end - timedelta(days=int(EXPORT_YEARS * 365))

    mt5_mod.symbol_select(symbol, True)
    time.sleep(0.3)
    sinfo = mt5_mod.symbol_info(symbol)
    if sinfo is None:
        logger.warning(f"  {symbol} {tf_name}: symbol_info failed")
        return None, None

    point = sinfo.point
    tick_value = sinfo.trade_tick_value
    volume_step = sinfo.volume_step if sinfo.volume_step > 0 else 0.01

    if tf_config["use_paged"]:
        sys.path.insert(0, str(BASE_DIR / "bot"))
        from mt5_connect import fetch_rates_paged
        df = fetch_rates_paged(symbol, tf, start, end)
        if df is None or len(df) < 100:
            logger.warning(f"  {symbol} {tf_name}: insufficient paged data ({len(df) if df is not None else 0})")
            return None, None
    else:
        rates = mt5_mod.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 100:
            logger.warning(f"  {symbol} {tf_name}: insufficient data ({len(rates) if rates is not None else 0})")
            return None, None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

    df = df.sort_values("time").reset_index(drop=True)
    if "spread" not in df.columns:
        df["spread"] = 0

    out = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].copy()
    out["time"] = out["time"].astype("datetime64[ns]").astype("int64") // 10**9

    csv_path = HISTORY_DIR / f"{symbol.replace('.', '_')}_{tf_name}.csv"
    out.to_csv(csv_path, index=False)
    logger.info(f"  SAVED {csv_path} ({len(out)} bars, {df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    return len(out), {"point": point, "tick_value": tick_value, "volume_step": volume_step}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export MT5 history to data/history/ CSVs")
    parser.add_argument("--tf", default="", help="Comma-separated timeframes to export (default: all)")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols (default: portfolio)")
    parser.add_argument("--no-git", action="store_true", help="Skip the git commit/push step")
    args = parser.parse_args()

    tf_map = TF_MAP
    if args.tf:
        wanted = {t.strip().upper() for t in args.tf.split(",") if t.strip()}
        tf_map = {k: v for k, v in TF_MAP.items() if k in wanted}
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or SYMBOLS

    logger = _setup_logging()
    logger.info("=" * 60)
    logger.info("MT5 Data Export Started")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {list(tf_map)} | years={EXPORT_YEARS}")
    logger.info("=" * 60)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    settings = _load_settings()

    # Connect via mt5linux (Linux NUC) or native (Windows)
    sys.path.insert(0, str(BASE_DIR / "bot"))
    try:
        from config import load_config  # noqa: I001
        from mt5_connect import ensure_mt5_connected, mt5 as mt5_mod  # noqa: I001
        cfg = load_config()
        cfg["symbols"] = symbols
        ok = ensure_mt5_connected(cfg)
        if not ok:
            logger.error("MT5 not connected — skipping export (timer will retry next cycle)")
            return
    except Exception as e:
        logger.error(f"MT5 connection failed: {e} — skipping export")
        return

    results = {}
    for symbol in symbols:
        results[symbol] = {}
        for tf_name, tf_config in tf_map.items():
            try:
                rows, info = _export_tf(mt5_mod, symbol, tf_name, tf_config, logger)
                if rows is not None and info is not None:
                    results[symbol][tf_name] = rows
                    # Persist point/tick once per symbol (from H1 pass)
                    if tf_name == "H1":
                        _save_symbol_info(settings, symbol, info["point"], info["tick_value"], info["volume_step"])
            except Exception as e:
                logger.warning(f"  {symbol} {tf_name}: export failed: {e}")
                continue

    # Git commit + push
    export_meta = {
        "timestamp": datetime.now().isoformat(),
        "symbols": {sym: tfs for sym, tfs in results.items() if tfs},
    }
    meta_path = HISTORY_DIR / ".export_meta.json"
    meta_path.write_text(json.dumps(export_meta, indent=2))

    if args.no_git:
        logger.info("--no-git: skipping commit/push")
        return

    try:
        subprocess.run(["git", "add", "data/history/", ".gitattributes", "config/settings.ini"],
                       cwd=BASE_DIR, check=True, capture_output=True)
        # Also stage .gitignore if changed
        subprocess.run(["git", "add", ".gitignore"], cwd=BASE_DIR, capture_output=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=BASE_DIR,
                                capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m", f"chore: MT5 data export {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
                cwd=BASE_DIR, check=True, capture_output=True,
            )
            subprocess.run(["git", "push"], cwd=BASE_DIR, check=True, capture_output=True)
            logger.info("Git commit + push successful")
        else:
            logger.info("No changes to commit")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e.stderr.decode() if hasattr(e.stderr, 'decode') else e}")

    total_files = sum(len(tfs) for tfs in results.values())
    logger.info(f"Export complete: {total_files} files across {len(SYMBOLS)} symbols")


if __name__ == "__main__":
    main()
