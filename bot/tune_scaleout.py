"""Grid-search scale-out and chandelier parameters per symbol."""

import configparser
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOGS_DIR = BASE_DIR / "logs"

sys.path.insert(0, str(BASE_DIR / "bot"))
from backtest import Backtest  # noqa: E402
from mt5_connect import login_account  # noqa: E402

SYMBOLS = ["ETHUSD.raw", "XAUUSD.raw", "XAGUSD.raw", "US30.raw", "EURUSD.raw"]
YEARS = 3

PARAM_GRID = {
    "tp_targets_rr": [[0.50, 0.75], [0.40, 0.60], [0.60, 0.85], [0.50, 0.70], [0.35, 0.55]],
    "chandelier_mult": [2.5, 3.0, 3.5],
    "chandelier_mult_partial": [1.5, 2.0],
}

BASE = {
    "ema_fast": 50,
    "ema_slow": 200,
    "atr_period": 14,
    "atr_sl_mult": 1.0,
    "rr": 2.0,
    "adx_enabled": True,
    "adx_trend": 25,
    "adx_range": 20,
    "stops_level": 50,
    "ml_confidence": 0.40,
    "ml_threshold_overrides": {},
    "volume_filter": True,
    "volume_kappa": 1.2,
    "volatility_filter": True,
    "atr_sma_period": 20,
    "chandelier_enabled": True,
    "chandelier_mult_overrides": {},
    "chandelier_lookback": 14,
    "ch_two_stage": True,
    "ch_loose_mult": 3.5,
    "ch_tight_mult": 1.5,
    "scale_out_enabled": True,
    "ml_enabled": True,
    "point": 0.01,
    "tick_value": 0.01,
    "risk_percent": 1.0,
    "initial_balance": 400000.0,
    "spread_model": 0.0,
    "volume_step": 0.01,
}

SYMBOL_OVERRIDE_MAP = {
    "ema_fast_period": ("ema_fast", int),
    "ema_slow_period": ("ema_slow", int),
    "atr_sl_multiplier": ("atr_sl_mult", float),
    "risk_reward_ratio": ("rr", float),
    "atr_sma_period": ("atr_sma_period", int),
    "risk_percent": ("risk_percent", float),
    "max_positions_per_symbol": ("max_positions_per_symbol", int),
}


def init_mt5():
    """Initialize MT5 and log in with the configured credentials."""
    if not login_account():
        print("MT5 init/login failed — see log for details")
        return False
    return True


def fetch_symbol_data(symbol, tf=mt5.TIMEFRAME_H1):
    mt5.symbol_select(symbol, True)
    end = datetime.now()
    start = end - pd.Timedelta(days=int(YEARS * 365))
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) < 200:
        print(f"Cannot fetch enough data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    sinfo = mt5.symbol_info(symbol)
    point = sinfo.point if sinfo else 0.01
    tick_value = sinfo.trade_tick_value if sinfo else 0.01
    volume_step = sinfo.volume_step if sinfo and sinfo.volume_step > 0 else 0.01
    model_path = BASE_DIR / "models" / f"model_{symbol.replace('.', '_')}.pkl"
    ml_path = str(model_path) if model_path.exists() else None
    return df, point, tick_value, volume_step, ml_path


def tune_symbol(symbol, data):
    import logging

    logging.getLogger().setLevel(logging.WARNING)

    print(f"\n{'=' * 60}")
    print(f"TUNING {symbol}")
    print(f"{'=' * 60}")

    df, point, tick_value, volume_step, model_path = data

    list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total = len(combos)
    print(f"Testing {total} combinations...")

    results = []
    for idx, values in enumerate(combos):
        params = dict(BASE)
        params.update(
            {
                "point": point,
                "tick_value": tick_value,
                "volume_step": volume_step,
                "scale_out_close_fractions": [0.30, 0.30],
                "scale_out_tp_targets_rr": values[0],
                "chandelier_mult": values[1],
                "chandelier_mult_partial": values[2],
            }
        )
        if model_path:
            params["ml_model_path"] = model_path

        settings = configparser.ConfigParser()
        settings.read(CONFIG_DIR / "settings.ini")
        sym_section = f"STRATEGY:{symbol}"
        if settings.has_section(sym_section):
            for ini_key, (cfg_key, converter) in SYMBOL_OVERRIDE_MAP.items():
                if settings.has_option(sym_section, ini_key):
                    params[cfg_key] = converter(settings.get(sym_section, ini_key))

        bt = Backtest(df, params)
        import contextlib
        import os

        with contextlib.redirect_stdout(open(os.devnull, "w")):
            bt.run()
        stats = bt.stats
        sc_trades = sum(1 for t in bt.trades if t.get("exit_reason") == "SCALE_OUT")
        ch_trades = sum(1 for t in bt.trades if t.get("exit_reason") == "CHANDELIER")
        results.append(
            {
                "tp_targets_rr": values[0],
                "ch_mult": values[1],
                "ch_mult_partial": values[2],
                "trades": stats["trades"],
                "return": round(stats["return"], 2),
                "max_dd": round(stats["max_dd"], 2),
                "win_rate": round(stats["win_rate"] * 100, 1),
                "profit_factor": round(stats["profit_factor"], 2),
                "sharpe": round(stats["sharpe"], 3),
                "calmar": round(stats["calmar"], 3),
                "scale_out": sc_trades,
                "chandelier": ch_trades,
            }
        )

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  {idx + 1}/{total} complete", flush=True)

    results.sort(key=lambda r: r["calmar"], reverse=True)
    print(f"\nTop 10 for {symbol}:")
    print(
        f"{'Rank':<5} {'RRfracs':<14} {'ChM':<5} {'ChP':<5} {'Trds':<5} "
        f"{'Return':<10} {'DD':<10} {'WR':<5} {'PF':<5} {'Sharpe':<8} "
        f"{'Calmar':<8} {'SCO':<4} {'CH':<4}"
    )
    for i, r in enumerate(results[:10]):
        print(
            f"{i + 1:<5} {str(r['tp_targets_rr']):<14} {r['ch_mult']:<5} {r['ch_mult_partial']:<5} "
            f"{r['trades']:<5} {r['return']:<10} {r['max_dd']:<10} {r['win_rate']:<5} "
            f"{r['profit_factor']:<5} {r['sharpe']:<8} {r['calmar']:<8} "
            f"{r['scale_out']:<4} {r['chandelier']:<4}"
        )

    report = {"symbol": symbol, "top10": results[:10], "all": results}
    report_path = LOGS_DIR / f"tune_{symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to {report_path}")
    return report


if __name__ == "__main__":
    print("Initializing MT5...")
    if not init_mt5():
        sys.exit(1)
    print("Fetching data for all symbols...")
    symbol_data = {}
    for symbol in SYMBOLS:
        print(f"  {symbol}...", end=" ", flush=True)
        result = fetch_symbol_data(symbol)
        if result:
            symbol_data[symbol] = result
            print(f"{len(result[0])} bars")
        else:
            print("FAILED")
    mt5.shutdown()

    for symbol, data in symbol_data.items():
        tune_symbol(symbol, data)
