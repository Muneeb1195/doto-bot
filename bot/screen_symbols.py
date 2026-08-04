"""Quick symbol screening — single-window backtest across all .raw symbols."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import configparser
import csv
import logging
import os
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
from backtest import Backtest

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

SCREEN_PROFILES = [
    {"name": "A", "ema_fast": 10, "ema_slow": 40, "atr_sl_mult": 1.5, "rr": 2.0, "adx_trend_threshold": 25},
    {"name": "B", "ema_fast": 8, "ema_slow": 34, "atr_sl_mult": 1.5, "rr": 2.0, "adx_trend_threshold": 25},
    {"name": "C", "ema_fast": 15, "ema_slow": 60, "atr_sl_mult": 1.5, "rr": 2.0, "adx_trend_threshold": 25},
    {"name": "D", "ema_fast": 12, "ema_slow": 48, "atr_sl_mult": 1.5, "rr": 2.0, "adx_trend_threshold": 25},
    {"name": "FAST", "ema_fast": 5, "ema_slow": 20, "atr_sl_mult": 1.0, "rr": 3.0, "adx_trend_threshold": 25},
]


def base_params(symbol, sinfo, profile):
    point = sinfo.point if sinfo else 0.01
    tick_value = sinfo.trade_tick_value if sinfo else 0.01
    volume_step = sinfo.volume_step if sinfo and sinfo.volume_step > 0 else 0.01
    return {
        "symbol": symbol,
        "timeframe": mt5.TIMEFRAME_H1,
        "ema_fast": profile["ema_fast"],
        "ema_slow": profile["ema_slow"],
        "atr_sl_mult": profile["atr_sl_mult"],
        "rr": profile["rr"],
        "adx_trend_threshold": profile["adx_trend_threshold"],
        "adx_trend_period": 14,
        "risk_percent": 1.0,
        "initial_balance": 500000.0,
        "max_positions_per_symbol": 1,
        "max_risk_ratio": 2.0,
        "point": point,
        "tick_value": tick_value,
        "volume_step": volume_step,
        "ml_enabled": False,
        "dr_enabled": False,
        "dr_vol_adjust": False,
        "spf_enabled": True,
        "spf_max_ratio": 0.30,
        "chandelier_enabled": True,
        "chandelier_mult": 3.0,
        "chandelier_mult_partial": 1.5,
        "chandelier_lookback": 14,
        "ch_two_stage": True,
        "ch_loose_mult": 3.5,
        "ch_tight_mult": 1.5,
        "ma_type": "kama",
        "scale_out_enabled": True,
        "scale_out_close_fractions": [0.20, 0.20],
        "scale_out_tp_targets_rr": [0.50, 0.75],
        "pb_enabled": True,
        "pb_atr_mult": 2.0,
        "mr_enabled": False,
        "tr_enabled": True,
        "tr_sigma": 3.0,
        "tr_lookback": 50,
        "tr_max_dd_pct": 8.0,
        "cb_dd_pct": 15.0,
        "daily_loss_pct": 5.0,
        "spread_model": 0.0,
    }


def screen():
    creds = configparser.ConfigParser()
    creds.read(CONFIG_DIR / "credentials.ini")
    account = int(os.getenv("MT5_ACCOUNT", creds["LOGIN"]["account"]))
    password = os.getenv("MT5_PASSWORD", creds["LOGIN"]["password"])
    server = os.getenv("MT5_SERVER", creds["LOGIN"]["server"])

    if not mt5.initialize(login=account, password=password, server=server):
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    all_symbols = mt5.symbols_get()
    raw_symbols = sorted([s.name for s in all_symbols if ".raw" in s.name])
    print(f"Found {len(raw_symbols)} .raw symbols")

    end = datetime.now()
    start = end - timedelta(days=730)  # ~2 years
    tf = mt5.TIMEFRAME_H1

    rows = []
    for symbol in raw_symbols:
        mt5.symbol_select(symbol, True)
        sinfo = mt5.symbol_info(symbol)
        if not sinfo:
            continue
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is None or len(rates) < 200:
            print(f"  {symbol}: insufficient data ({len(rates) if rates is not None else 0} bars)")
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")

        best_pf = 0
        best_row = None
        for profile in SCREEN_PROFILES:
            try:
                p = base_params(symbol, sinfo, profile)
                bt = Backtest(df.copy(), p)
                result = bt.run()
                pf = result.get("profit_factor", 0)
                wr = result.get("win_rate", 0)
                ret = result.get("total_return", 0)
                dd = result.get("max_dd", 0)
                trades = result.get("n_trades", 0)
                if pf > best_pf:
                    best_pf = pf
                    best_row = {
                        "symbol": symbol,
                        "profile": profile["name"],
                        "ema_fast": profile["ema_fast"],
                        "ema_slow": profile["ema_slow"],
                        "pf": round(pf, 4),
                        "wr": round(wr, 4),
                        "ret": round(ret, 2),
                        "dd": round(dd, 2),
                        "trades": trades,
                        "bars": len(df),
                    }
            except Exception as e:
                print(f"  {symbol} {profile['name']} error: {e}")
                continue

        if best_row:
            rows.append(best_row)
            print(
                f"  {symbol}: best={best_row['profile']} PF={best_row['pf']} "
                f"WR={best_row['wr']} Ret={best_row['ret']} Trades={best_row['trades']}"
            )

    mt5.shutdown()

    rows.sort(key=lambda r: r["pf"], reverse=True)
    out_path = LOG_DIR / "symbol_screen.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["symbol", "profile", "ema_fast", "ema_slow", "pf", "wr", "ret", "dd", "trades", "bars"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {out_path}")
    print("\nTop 10 by PF:")
    for r in rows[:10]:
        print(
            f"  {r['symbol']:20s} {r['profile']:5s} PF={r['pf']:.4f} WR={r['wr']:.4f} "
            f"Ret={r['ret']:>8.0f} DD={r['dd']:.0f} Trades={r['trades']}"
        )


if __name__ == "__main__":
    screen()
