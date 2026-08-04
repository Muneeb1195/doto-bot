"""Parallel walk-forward optimizer — fetch all data, then run all backtests in parallel."""

import argparse
import configparser
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from backtest import Backtest

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
HISTORY_DIR = DATA_DIR / "history"

settings = configparser.ConfigParser()
settings.read(CONFIG_DIR / "settings.ini")
creds = configparser.ConfigParser()
creds.read(CONFIG_DIR / "credentials.ini")

SYMBOL_PROFILE = {
    # --- Live portfolio (7 symbols) ---
    "XAU500.raw": "D",
    "BTCUSD.raw": "FAST",
    "US30.raw": "C",
    "GBPJPY.raw": "B",
    "SOLUSD.raw": "FAST",
    "XRPUSD.raw": "FAST",
    "DOGUSD.raw": "FAST",
    # --- Pool members with trained models ---
    "EURUSD.raw": "A",
    "NZDUSD.raw": "A",
    "USDJPY.raw": "B",
    "EURJPY.raw": "B",
    "ETHUSD.raw": "C",
    "US500.raw": "C",
    "LTCUSD.raw": "FAST",
    "ADAUSD.raw": "FAST",
    "AVXUSD.raw": "FAST",
    "GBPUSD.raw": "A",
    "AUDUSD.raw": "A",
    "XAUUSD.raw": "D",
    "XAGUSD.raw": "D",
    "XNGUSD.raw": "C",
    "XPTUSD.raw": "D",
    "SPY.raw": "C",
    "IWM.raw": "C",
    "USDCHF.raw": "A",
    "USDCAD.raw": "A",
}

PROFILE_EMAS = {
    "A": [(10, 40), (12, 48), (15, 60)],
    "B": [(8, 34), (10, 40)],
    "C": [(12, 48), (15, 60), (20, 80)],
    "D": [(10, 40), (12, 48)],
    "FAST": [(5, 20), (8, 24), (10, 30)],
}
PROFILE_MR = {"A": True, "B": False, "C": False, "D": False, "FAST": True}
PROFILE_SL = {"A": [1.0, 1.5, 2.0], "B": [1.0, 1.5, 2.0], "C": [1.0, 1.5, 2.0], "D": [1.5, 2.0], "FAST": [1.0, 1.5]}
PROFILE_RR = {"A": [1.5, 2.0], "B": [2.0, 2.5], "C": [2.0, 2.5], "D": [2.0, 2.5, 5.0], "FAST": [2.0, 3.0]}
PROFILE_ADX = {"A": [22, 25], "B": [22, 25], "C": [22, 25], "D": [25], "FAST": [25, 30]}
PROFILE_COMMISSION = {"A": 976.0, "B": 976.0, "C": 278.0, "D": 977.0, "FAST": 558.0}


def connect_mt5():
    mt5_path = settings.get("MT5", "path", fallback="C:\\Program Files\\MetaTrader 5\\terminal64.exe")
    timeout_ms = settings.getint("MT5", "timeout_ms", fallback=180000)
    import platform
    import subprocess

    import MetaTrader5 as mt5

    ok = mt5.initialize(
        path=mt5_path,
        login=int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
        password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
        server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
        timeout=timeout_ms,
    )
    if not ok:
        print(f"MT5 init failed ({mt5.last_error()}) - restarting...")
        if platform.system() == "Linux":
            subprocess.run(["pkill", "-f", "terminal64.exe"], capture_output=True)
            subprocess.run(["pkill", "-f", "winedevice.exe"], capture_output=True)
        else:
            subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "winedevice.exe"], capture_output=True)
        time.sleep(5)
        if platform.system() == "Linux":
            subprocess.Popen(["wine", mt5_path])
        else:
            subprocess.Popen([mt5_path])
        time.sleep(15)
        ok = mt5.initialize(
            path=mt5_path,
            login=int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
            password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
            server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
            timeout=timeout_ms,
        )
    if not ok:
        print(f"MT5 re-init failed: {mt5.last_error()}")
        return None
    authorized = mt5.login(
        int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
        password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
        server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
    )
    if not authorized:
        print(f"MT5 login failed (continuing anyway): {mt5.last_error()}")
    return mt5


def load_csv_data(symbol):
    """Load pre-fetched H1 bars from data/history/<SYMBOL>.csv.

    Expected columns: time,open,high,low,close,tick_volume[,spread].
    `time` may be epoch seconds or an ISO/datetime string. Returns the same
    (df, info) tuple as fetch_data so the optimizer is agnostic to the source.
    """
    csv_path = HISTORY_DIR / f"{symbol.replace('.', '_')}.csv"
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return None, None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        print(f"  CSV missing 'time' column: {csv_path}")
        return None, None
    df["time"].iloc[0]
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s")
    else:
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    if "spread" not in df.columns:
        df["spread"] = 0
    info = {
        "point": float(settings.get("SYMBOL_POINTS", symbol, fallback=0.01))
        if settings.has_section("SYMBOL_POINTS") and settings.has_option("SYMBOL_POINTS", symbol)
        else 0.01,
        "tick_value": float(settings.get("SYMBOL_POINTS", symbol + "_tick", fallback=1.0))
        if settings.has_section("SYMBOL_POINTS") and settings.has_option("SYMBOL_POINTS", symbol + "_tick")
        else 1.0,
        "volume_step": 0.01,
    }
    # Allow explicit point/tick override via settings [SYMBOL_POINTS].
    if settings.has_section("SYMBOL_POINTS"):
        if settings.has_option("SYMBOL_POINTS", symbol):
            info["point"] = float(settings.get("SYMBOL_POINTS", symbol))
        if settings.has_option("SYMBOL_POINTS", symbol + "_tick"):
            info["tick_value"] = float(settings.get("SYMBOL_POINTS", symbol + "_tick"))
        if settings.has_option("SYMBOL_POINTS", symbol + "_vstep"):
            info["volume_step"] = float(settings.get("SYMBOL_POINTS", symbol + "_vstep"))
    print(
        f"  {symbol}: {len(df)} bars from CSV ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}) "
        f"point={info['point']} tick_val={info['tick_value']}"
    )
    return df, info


def fetch_data(symbol, years=3, csv_only=False):
    if csv_only:
        return load_csv_data(symbol)
    import MetaTrader5 as mt5_mod

    mt5_mod.symbol_select(symbol, True)
    sinfo = mt5_mod.symbol_info(symbol)
    if not sinfo:
        return None, None
    point = sinfo.point
    tick_value = sinfo.trade_tick_value
    volume_step = sinfo.volume_step if sinfo.volume_step > 0 else 0.01
    if tick_value == 0:
        mt5_mod.symbol_select(symbol, False)
        time.sleep(0.3)
        mt5_mod.symbol_select(symbol, True)
        time.sleep(0.5)
        sinfo = mt5_mod.symbol_info(symbol)
        if sinfo:
            tick_value = sinfo.trade_tick_value
    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    rates = mt5_mod.copy_rates_range(symbol, mt5_mod.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) < 200:
        print(f"  {symbol}: insufficient data ({len(rates) if rates is not None else 0} bars)")
        return None, None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(
        f"  {symbol}: {len(df)} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()}) "
        f"point={point} tick_val={tick_value}"
    )
    acc_info = mt5_mod.account_info()
    if acc_info and acc_info.currency.upper() != "USD":
        base_cur = acc_info.currency.upper()
        fx_pair = f"{base_cur}USD"
        mt5_mod.symbol_select(fx_pair, True)
        fx_rates = mt5_mod.copy_rates_from_pos(fx_pair, mt5_mod.TIMEFRAME_H1, 0, 3)
        if fx_rates is not None and len(fx_rates) > 0:
            rate = float(fx_rates[-1]["close"])
            tick_value /= rate
    return df, {"point": point, "tick_value": tick_value, "volume_step": volume_step}


def make_windows(df, n_windows=4, window_months=24, oos_months=6):
    times = df["time"]
    end = times.iloc[-1]
    windows = []
    for i in range(n_windows):
        oos_end = end - pd.DateOffset(months=i * oos_months)
        oos_start = oos_end - pd.DateOffset(months=oos_months)
        is_end = oos_start
        is_start = is_end - pd.DateOffset(months=window_months)
        is_mask = (times >= is_start) & (times < is_end)
        oos_mask = (times >= oos_start) & (times < oos_end)
        df_is = df[is_mask].reset_index(drop=True)
        df_oos = df[oos_mask].reset_index(drop=True)
        if len(df_is) < 200 or len(df_oos) < 100:
            continue
        windows.append((df_is, df_oos, is_start, is_end, oos_start, oos_end))
    return windows


def _resolve_ma_type(symbol):
    sym_section = f"STRATEGY:{symbol}"
    if settings.has_section(sym_section) and settings.has_option(sym_section, "ma_type"):
        return settings.get(sym_section, "ma_type")
    if settings.has_option("STRATEGY", "ma_type"):
        return settings.get("STRATEGY", "ma_type")
    return "kama"


def build_params(
    symbol,
    ema_fast,
    ema_slow,
    sl,
    rr,
    adx,
    point,
    tick_value,
    volume_step,
    mr_enabled=True,
    profile="A",
    ma_type="kama",
):
    from optimize_params import build_params as _delegate

    commission = PROFILE_COMMISSION.get(profile, 0.0)
    return _delegate(
        symbol,
        ema_fast,
        ema_slow,
        sl,
        rr,
        adx,
        point,
        tick_value,
        volume_step,
        mr_enabled=mr_enabled,
        commission=commission,
        ma_type=ma_type,
        no_ml=False,
    )


def run_bt(df, params, df_m15=None, fast=False):
    bt = Backtest(df, params, df_m15=df_m15)
    bt.run(fast=fast)
    return bt.get_results()


def score_r(r):
    pf = r.get("profit_factor", 0) or 0
    ret = r.get("total_return", 0) or 0
    dd = r.get("max_dd", 10000) or 10000
    wr = r.get("win_rate", 0) or 0
    n = r.get("n_trades", 0)
    if n < 3:
        return -999
    if pf == float("inf") or pf > 10:
        pf = 10
    if pf > 3.0:
        pf = 3.0
    base = ret - dd * 0.3
    if base < 0 and pf > 1:
        base *= 0.5
    mult = pf
    wr_penalty = 1.0
    if wr > 0.60:
        wr_penalty = 1.0 - min(0.7, (wr - 0.60) * 5)
    if 0.50 <= wr <= 0.55:
        wr_penalty = 1.15
    pf_penalty = 1.0
    if pf > 2.0:
        pf_penalty = max(0.3, 1.0 - (pf - 2.0) * 0.3)
    if 1.2 <= pf <= 1.6:
        pf_penalty = 1.15
    return base * mult * wr_penalty * pf_penalty


def score_walk_forward(window_results):
    is_scores = []
    oos_pfs = []
    degradations = []
    for wr in window_results:
        is_r = wr["is"]
        oos_r = wr["oos"]
        is_s = score_r(is_r)
        is_scores.append(is_s)
        oos_pf = oos_r.get("profit_factor", 0) or 0
        oos_pfs.append(oos_pf)
        if is_s > 0 and oos_r.get("n_trades", 0) >= 3:
            oos_s = score_r(oos_r)
            deg = (is_s - oos_s) / max(abs(is_s), 0.01)
            degradations.append(deg)
    if not is_scores:
        return -999
    avg_is = float(np.mean(is_scores))
    deg_penalty = 1.0
    for deg in degradations:
        if deg > 0.50:
            deg_penalty *= max(0.1, 1.0 - (deg - 0.30) * 2)
    good_oos = sum(1 for pf in oos_pfs if pf >= 1.0)
    if good_oos >= 2:
        deg_penalty *= 1.15
    elif good_oos == 0:
        deg_penalty *= 0.5
    return avg_is * deg_penalty


def bt_task(args):
    df, params, df_m15, fast = args
    return run_bt(df, params, df_m15=df_m15, fast=fast)


def optimize_symbol_parallel(symbol, df_full, info, windows, max_workers=8, fast=False, df_m15=None):
    point = info["point"]
    tick_value = info["tick_value"]
    volume_step = info["volume_step"]
    profile = SYMBOL_PROFILE.get(symbol, "A")
    mr_enabled = PROFILE_MR[profile]
    ema_grid = PROFILE_EMAS[profile]
    sl_vals = PROFILE_SL[profile]
    rr_vals = PROFILE_RR[profile]
    adx_vals = PROFILE_ADX[profile]
    ma_type = _resolve_ma_type(symbol)

    print(f"\n{'=' * 70}")
    print(f"Profile {profile} — {symbol} (MR={'YES' if mr_enabled else 'NO'})")
    print(f"  Data: {len(df_full)} bars, {len(windows)} windows")
    print(f"  EMA={ema_grid} SL={sl_vals} RR={rr_vals} ADX={adx_vals}")
    print(f"{'=' * 70}")

    w1 = windows[0][0]
    print(f"  First IS: {w1['time'].iloc[0].date()} to {w1['time'].iloc[-1].date()}")

    n_phase1 = min(2, len(windows))

    print(f"\n  Phase 1: EMA sweep ({len(ema_grid)} EMAs × {n_phase1} windows)")
    phase1_tasks = []
    phase1_keys = []
    for ema_fast, ema_slow in ema_grid:
        for w_idx in range(n_phase1):
            params = build_params(
                symbol, ema_fast, ema_slow, 1.5, 2.0, 25, point, tick_value, volume_step, mr_enabled, profile, ma_type
            )
            phase1_tasks.append((windows[w_idx][0].copy(), params.copy(), df_m15, fast))
            phase1_keys.append((symbol, ema_fast, ema_slow, 1.5, 2.0, 25, w_idx, "is"))
            phase1_tasks.append((windows[w_idx][1].copy(), params.copy(), df_m15, fast))
            phase1_keys.append((symbol, ema_fast, ema_slow, 1.5, 2.0, 25, w_idx, "oos"))

    phase1_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(bt_task, t): k for t, k in zip(phase1_tasks, phase1_keys)}
        for f in as_completed(futures):
            k = futures[f]
            phase1_results[k] = f.result()

    ema_scores = {}
    for ema_fast, ema_slow in ema_grid:
        wr_list = []
        for w_idx in range(n_phase1):
            is_k = (symbol, ema_fast, ema_slow, 1.5, 2.0, 25, w_idx, "is")
            oos_k = (symbol, ema_fast, ema_slow, 1.5, 2.0, 25, w_idx, "oos")
            if is_k in phase1_results and oos_k in phase1_results:
                wr_list.append({"is": phase1_results[is_k], "oos": phase1_results[oos_k], "window": w_idx})
        if wr_list:
            ema_scores[(ema_fast, ema_slow)] = score_walk_forward(wr_list)

    ema_ranked = sorted(ema_scores.items(), key=lambda x: x[1], reverse=True)
    for (ef, es), sc in ema_ranked:
        print(f"    EMA{ef}/{es}: WF score={sc:.1f}")
    top_emas = [pair for pair, _ in ema_ranked[:2]]

    print(f"\n  Phase 2: SL/RR/ADX refinement ({len(windows)} windows)")
    phase2_tasks = []
    phase2_keys = []
    for ema_fast, ema_slow in top_emas:
        for sl in sl_vals:
            for rr in rr_vals:
                for w_idx in range(len(windows)):
                    params = build_params(
                        symbol,
                        ema_fast,
                        ema_slow,
                        sl,
                        rr,
                        25,
                        point,
                        tick_value,
                        volume_step,
                        mr_enabled,
                        profile,
                        ma_type,
                    )
                    phase2_tasks.append((windows[w_idx][0].copy(), params.copy(), df_m15, fast))
                    phase2_keys.append((symbol, ema_fast, ema_slow, sl, rr, 25, w_idx, "is"))
                    phase2_tasks.append((windows[w_idx][1].copy(), params.copy(), df_m15, fast))
                    phase2_keys.append((symbol, ema_fast, ema_slow, sl, rr, 25, w_idx, "oos"))
            for adx in adx_vals:
                for w_idx in range(len(windows)):
                    params = build_params(
                        symbol,
                        ema_fast,
                        ema_slow,
                        1.5,
                        2.0,
                        adx,
                        point,
                        tick_value,
                        volume_step,
                        mr_enabled,
                        profile,
                        ma_type,
                    )
                    phase2_tasks.append((windows[w_idx][0].copy(), params.copy(), df_m15, fast))
                    phase2_keys.append((symbol, ema_fast, ema_slow, 1.5, 2.0, adx, w_idx, "is"))
                    phase2_tasks.append((windows[w_idx][1].copy(), params.copy(), df_m15, fast))
                    phase2_keys.append((symbol, ema_fast, ema_slow, 1.5, 2.0, adx, w_idx, "oos"))

    phase2_results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(bt_task, t): k for t, k in zip(phase2_tasks, phase2_keys)}
        for f in as_completed(futures):
            k = futures[f]
            phase2_results[k] = f.result()

    all_results = {**phase1_results, **phase2_results}

    candidates = []
    for ema_fast, ema_slow in top_emas:
        for sl in sl_vals:
            for rr in rr_vals:
                wr_list = []
                for w_idx in range(len(windows)):
                    is_k = (symbol, ema_fast, ema_slow, sl, rr, 25, w_idx, "is")
                    oos_k = (symbol, ema_fast, ema_slow, sl, rr, 25, w_idx, "oos")
                    if is_k in all_results and oos_k in all_results:
                        wr_list.append({"is": all_results[is_k], "oos": all_results[oos_k], "window": w_idx})
                if wr_list:
                    wf_s = score_walk_forward(wr_list)
                    candidates.append((ema_fast, ema_slow, sl, rr, 25, wr_list, wf_s))
        for adx in adx_vals:
            wr_list = []
            for w_idx in range(len(windows)):
                is_k = (symbol, ema_fast, ema_slow, 1.5, 2.0, adx, w_idx, "is")
                oos_k = (symbol, ema_fast, ema_slow, 1.5, 2.0, adx, w_idx, "oos")
                if is_k in all_results and oos_k in all_results:
                    wr_list.append({"is": all_results[is_k], "oos": all_results[oos_k], "window": w_idx})
            if wr_list:
                wf_s = score_walk_forward(wr_list)
                candidates.append((ema_fast, ema_slow, 1.5, 2.0, adx, wr_list, wf_s))

    candidates.sort(key=lambda x: x[6], reverse=True)
    best = candidates[0]
    ef, es, sl, rr, adx, wf_r, wf_s = best
    wf_is_r = wf_r[-1]["is"]

    print(f"\n  BEST for {symbol}:")
    print(f"    EMA{ef}/{es} SL={sl:.1f} RR={rr:.1f} ADX={adx}  WF={wf_s:.1f}")
    print(
        f"    IS: PF={wf_is_r.get('profit_factor', 0):.2f} Ret=Rs.{wf_is_r.get('total_return', 0):+.1f} "
        f"DD=Rs.{wf_is_r.get('max_dd', 0):.1f} n={wf_is_r.get('n_trades', 0)} WR={wf_is_r.get('win_rate', 0):.1%}"
    )
    for w_idx, wr in enumerate(wf_r):
        oos = wr["oos"]
        print(
            f"    W{w_idx + 1} OOS: PF={oos.get('profit_factor', 0):.2f} "
            f"Ret=Rs.{oos.get('total_return', 0):+.1f} n={oos.get('n_trades', 0)} "
            f"WR={oos.get('win_rate', 0):.1%}"
        )

    rows = []
    for ef2, es2, sl2, rr2, adx2, wf_r2, wf_s2 in candidates:
        is_last = wf_r2[-1]["is"]
        rows.append(
            {
                "ema_fast": ef2,
                "ema_slow": es2,
                "sl": sl2,
                "rr": rr2,
                "adx": adx2,
                "pf": is_last.get("profit_factor", 0),
                "ret": is_last.get("total_return", 0),
                "dd": is_last.get("max_dd", 0),
                "wr": is_last.get("win_rate", 0),
                "n_trades": is_last.get("n_trades", 0),
                "wf_score": wf_s2,
            }
        )
    pd.DataFrame(rows).to_csv(LOG_DIR / f"optimize_{symbol.replace('.', '_')}.csv", index=False)

    return {
        "symbol": symbol,
        "ema_fast": ef,
        "ema_slow": es,
        "sl": sl,
        "rr": rr,
        "adx": adx,
        "pf": wf_is_r.get("profit_factor", 0),
        "ret": wf_is_r.get("total_return", 0),
        "dd": wf_is_r.get("max_dd", 0),
        "wr": wf_is_r.get("win_rate", 0),
        "n_trades": wf_is_r.get("n_trades", 0),
        "score": wf_s,
        "n_windows": len(windows),
        "oos_pfs": [wr["oos"].get("profit_factor", 0) for wr in wf_r],
    }


def _fetch_csv_mode(args):
    """One-time data harvest: pull H1 bars from MT5 and save to data/history/.

    Requires the live MT5 terminal (credentials + login). After this runs once,
    subsequent optimization runs can use --csv (offline, bit-exact fast path).
    """
    print("Fetch-CSV mode: pulling H1 bars from MT5 and saving to data/history/")
    mt5 = connect_mt5()
    if mt5 is None:
        return
    import MetaTrader5 as mt5_mod

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if args.symbols and args.symbols.upper() != "ALL":
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = [
            "EURUSD.raw",
            "NZDUSD.raw",
            "USDJPY.raw",
            "EURJPY.raw",
            "GBPJPY.raw",
            "ETHUSD.raw",
            "US500.raw",
            "XAU500.raw",
            "DOGUSD.raw",
            "LTCUSD.raw",
            "BTCUSD.raw",
            "AVGO.raw",
            "ADBE.raw",
            "AMGN.raw",
        ]

    saved = 0
    for symbol in symbols:
        df, info = fetch_data(symbol, args.years, csv_only=False)
        if df is None or len(df) < 500:
            print(f"  SKIP {symbol} (no data)")
            continue
        out = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].copy()
        # Upcast to nanosecond epoch, then store as epoch seconds.
        out["time"] = out["time"].astype("datetime64[ns]").astype("int64") // 10**9  # epoch seconds
        csv_path = HISTORY_DIR / f"{symbol.replace('.', '_')}.csv"
        out.to_csv(csv_path, index=False)
        # Persist point/tick_value so offline runs can reconstruct PnL scale.
        if not settings.has_section("SYMBOL_POINTS"):
            settings.add_section("SYMBOL_POINTS")
        settings.set("SYMBOL_POINTS", symbol, str(info["point"]))
        settings.set("SYMBOL_POINTS", symbol + "_tick", str(info["tick_value"]))
        settings.set("SYMBOL_POINTS", symbol + "_vstep", str(info["volume_step"]))
        with open(CONFIG_DIR / "settings.ini", "w") as fh:
            settings.write(fh)
        print(f"  SAVED {csv_path} ({len(out)} bars)")
        saved += 1

    mt5_mod.shutdown()
    print(f"\nSaved {saved}/{len(symbols)} symbols to {HISTORY_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Parallel walk-forward optimizer")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--years", type=float, default=3)
    parser.add_argument("--threads", type=int, default=16, help="Parallel backtest threads")
    parser.add_argument(
        "--fast", action="store_true", help="Use the bit-exact Numba fast path (backtest.run(fast=True))"
    )
    parser.add_argument("--csv", action="store_true", help="Load H1 bars from data/history/<SYMBOL>.csv instead of MT5")
    parser.add_argument(
        "--fetch-csv",
        action="store_true",
        help="One-time: pull H1 bars from MT5 and save to data/history/<SYMBOL>.csv, then exit",
    )
    args = parser.parse_args()

    if args.fetch_csv:
        _fetch_csv_mode(args)
        return

    if args.csv:
        print("CSV mode: loading bars from data/history/*.csv (no MT5 required)")
    else:
        print("Connecting to MT5...", flush=True)
        mt5 = connect_mt5()
        if mt5 is None:
            return
        import MetaTrader5 as mt5_mod

    if args.symbols and args.symbols.upper() != "ALL":
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        # All 14 for final comparison
        symbols = [
            "EURUSD.raw",
            "NZDUSD.raw",
            "USDJPY.raw",
            "EURJPY.raw",
            "GBPJPY.raw",
            "ETHUSD.raw",
            "US500.raw",
            "XAU500.raw",
            "DOGUSD.raw",
            "LTCUSD.raw",
            "BTCUSD.raw",
            "AVGO.raw",
            "ADBE.raw",
            "AMGN.raw",
        ]

    print(f"\n{'=' * 70}")
    print(f"Phase 0: Fetching data for {len(symbols)} symbols")
    print(f"{'=' * 70}")
    all_data = {}
    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        # Fetch M15 BEFORE H1 (MT5 Python API bug: H1 fetch corrupts subsequent
        # M15 fetches). M15 drives the MTF entry path in the backtest — without
        # it, MTF-enabled symbols are evaluated on a degraded signal path.
        df_m15 = None
        df_m1 = None
        if not args.csv:
            from optimize_params import fetch_m1_data, fetch_m15_data

            df_m15 = fetch_m15_data(symbol, args.years)
            df_m1 = fetch_m1_data(symbol, args.years)
        data = fetch_data(symbol, args.years, csv_only=args.csv)
        if data is None or data[0] is None:
            print(f"  SKIP {symbol}")
            continue
        df, info = data
        if len(df) < 5000:
            print(f"  Too few bars ({len(df)}), SKIP")
            continue
        # Attach of_* orderflow features to the H1 frame before window slicing
        # (ML-gate parity with training/live post-item-#11).
        if df_m1 is not None:
            from ml_features import attach_orderflow_features

            attach_orderflow_features(df, df_m1)
            print(f"  M1: {len(df_m1)} bars (of_* attached)")
        print(f"  M15: {len(df_m15) if df_m15 is not None else 0} bars for MTF entry TF")
        windows = make_windows(df, n_windows=4, window_months=24, oos_months=6)
        if len(windows) < 2:
            print(f"  Only {len(windows)} windows (need >=2), SKIP")
            continue
        print(f"  Created {len(windows)} walk-forward windows")
        all_data[symbol] = (df, info, windows, df_m15)

    if not args.csv and "mt5_mod" in dir():
        mt5_mod.shutdown()
    print(f"\nSuccessfully fetched data for {len(all_data)}/{len(symbols)} symbols")

    if not all_data:
        print("No symbols to optimize.")
        return

    print(f"\n{'=' * 70}")
    print(f"Phase 1+2: Running parallel backtests for {len(all_data)} symbols ({args.threads} processes)")
    print(f"{'=' * 70}")

    all_best = []
    for symbol, (df, info, windows, df_m15) in all_data.items():
        best = optimize_symbol_parallel(
            symbol, df, info, windows, max_workers=args.threads, fast=args.fast, df_m15=df_m15
        )
        if best:
            all_best.append(best)

    print(f"\n{'=' * 80}")
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print(f"{'=' * 80}")
    for b in sorted(all_best, key=lambda x: x["score"], reverse=True):
        oos_str = " ".join(f"PF={pf:.2f}" for pf in b.get("oos_pfs", []))
        print(
            f"  {b['symbol']:15s} EMA{b['ema_fast']:2d}/{b['ema_slow']:<3d} "
            f"SL={b['sl']:.1f} RR={b['rr']:.1f} ADX={b['adx']:2d}  "
            f"PF={b['pf']:.2f} Ret=Rs.{b['ret']:+.1f} DD=Rs.{b['dd']:.1f} "
            f"WR={b.get('wr', 0):.1%} n={b['n_trades']:2d}  "
            f"WF={b['score']:.0f} windows={b['n_windows']}  [{oos_str}]"
        )


if __name__ == "__main__":
    main()
