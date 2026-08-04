"""validate_entry_config.py — pre-flight sanity check before a CPCC optimization run.

Run this before `parallel_optimize.py --fast --csv` to abort early (non-zero
exit) if the config is inconsistent, missing ML models, or has contradictory
flags. Keeps a long multi-symbol grid run from failing midway.

Usage:
    python -m bot.validate_entry_config [--symbols X,Y,Z] [--min-entry-score 0.60]
"""

import argparse
import configparser
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"

VALID_MA_TYPES = {"kama", "vidya", "ema"}
VALID_REGIMES = {"kama", "vidya"}  # the two MA dispatchers the bot actually uses


def err(msg):
    print(f"[FAIL] {msg}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate settings.ini before optimization")
    parser.add_argument(
        "--symbols", type=str, default=None, help="Comma-separated symbols to validate (default: [PORTFOLIO] symbols)"
    )
    parser.add_argument(
        "--min-entry-score", type=float, default=0.60, help="Required SCORING min_entry_score (Phase D target)"
    )
    args = parser.parse_args()

    settings = configparser.ConfigParser()
    ini_path = CONFIG_DIR / "settings.ini"
    if not ini_path.exists():
        print(f"[FAIL] settings.ini not found at {ini_path}")
        return 2
    settings.read(ini_path)

    ok = True

    # Resolve symbol list.
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif settings.has_section("PORTFOLIO") and settings.has_option("PORTFOLIO", "symbols"):
        raw = settings.get("PORTFOLIO", "symbols")
        symbols = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        ok = err("No symbols: pass --symbols or set [PORTFOLIO] symbols")
        symbols = []

    if not symbols:
        ok = err("Symbol list is empty") and ok

    # [TRADING] basics.
    if not settings.has_section("TRADING"):
        ok = err("[TRADING] section missing") and ok
    else:
        rp = settings.getfloat("TRADING", "risk_percent", fallback=None)
        if rp is None or not (0 < rp <= 5):
            ok = err(f"TRADING.risk_percent must be in (0, 5]; got {rp}") and ok
        dll = settings.getfloat("TRADING", "daily_loss_limit_percent", fallback=None)
        if dll is None or not (0 < dll <= 50):
            ok = err(f"TRADING.daily_loss_limit_percent must be in (0, 50]; got {dll}") and ok

    # [SCORING] min_entry_score == target.
    if settings.has_section("SCORING") and settings.has_option("SCORING", "min_entry_score"):
        mes = settings.getfloat("SCORING", "min_entry_score")
        if abs(mes - args.min_entry_score) > 1e-9:
            ok = err(f"SCORING.min_entry_score={mes} != required {args.min_entry_score}") and ok
    else:
        ok = err("[SCORING] min_entry_score missing") and ok

    # [ADX] thresholds sane.
    if settings.has_section("ADX"):
        at = settings.getfloat("ADX", "adx_trend_threshold", fallback=None)
        ar = settings.getfloat("ADX", "adx_range_threshold", fallback=None)
        if at is not None and ar is not None and not (0 < ar < at):
            ok = err(f"ADX: adx_range_threshold ({ar}) must be < adx_trend_threshold ({at})") and ok

    # Per-symbol strategy sections.
    for sym in symbols:
        sec = f"STRATEGY:{sym}"
        if not settings.has_section(sec):
            ok = err(f"Missing [STRATEGY:{sym}] section") and ok
            continue
        ma = settings.get(sec, "ma_type", fallback=None) or settings.get("STRATEGY", "ma_type", fallback=None)
        if ma not in VALID_MA_TYPES:
            ok = err(f"{sym}: invalid ma_type '{ma}' (expected one of {sorted(VALID_MA_TYPES)})") and ok
        ef = settings.getint(sec, "ema_fast_period", fallback=None)
        es = settings.getint(sec, "ema_slow_period", fallback=None)
        if ef is None or es is None or not (0 < ef < es):
            ok = err(f"{sym}: need 0 < ema_fast_period ({ef}) < ema_slow_period ({es})") and ok
        atr = settings.getfloat(sec, "atr_sl_multiplier", fallback=None)
        if atr is not None and not (0 < atr <= 5):
            ok = err(f"{sym}: atr_sl_multiplier {atr} out of (0, 5]") and ok
        rr = settings.getfloat(sec, "risk_reward_ratio", fallback=None)
        if rr is not None and not (0 < rr <= 10):
            ok = err(f"{sym}: risk_reward_ratio {rr} out of (0, 10]") and ok

        # ML model file present.
        model_path = MODELS_DIR / f"model_{sym.replace('.', '_')}.pkl"
        if not model_path.exists():
            ok = err(f"{sym}: ML model missing at {model_path}") and ok

    if ok:
        print(f"[OK] settings.ini valid for {len(symbols)} symbols: {', '.join(symbols)}")
        return 0
    print("[FAIL] configuration invalid — aborting before optimization")
    return 1


if __name__ == "__main__":
    sys.exit(main())
