import configparser
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
from indicators import calc_atr_series, calc_ma

_OPT_POOL = ThreadPoolExecutor(max_workers=1)


def _mt5_call_timeout(func, *args, _timeout=30, **kwargs):
    future = _OPT_POOL.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=_timeout)
    except FutureTimeoutError:
        logging.warning(f"MT5 {func.__name__} timed out ({_timeout}s)")
        return None


import mc_validation  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

PARAM_GRID = {
    "ema_fast": [30, 50, 70],
    "ema_slow": [100, 150, 200],
    "atr_sl_mult": [1.0, 1.2, 1.5],
    "rr": [1.5, 2.0, 2.5],
    "adx_trend_threshold": [20, 25, 30],
}

BAR_COUNT = 10000


def load_creds():
    creds = configparser.ConfigParser()
    creds.read(CONFIG_DIR / "credentials.ini")
    return {
        "server": os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
        "account": int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
        "password": os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
    }


def init_mt5():
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")
    mt5_path = settings["MT5"]["path"]
    timeout = int(settings["MT5"]["timeout_ms"])
    if not mt5.initialize(path=mt5_path, timeout=timeout):
        logging.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    creds = load_creds()
    authorized = mt5.login(creds["account"], password=creds["password"], server=creds["server"])
    if not authorized:
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    ai = mt5.account_info()
    logging.info(f"Connected: {ai.name} | Balance: ${ai.balance:.2f}")
    return True


def fetch_data(symbol, timeframe, bars=BAR_COUNT):
    rates = _mt5_call_timeout(mt5.copy_rates_from_pos, symbol, timeframe, 0, bars, _timeout=60)
    if rates is None or (isinstance(rates, (list, tuple)) and len(rates) < 500):
        logging.error(f"Could not fetch {bars} bars for {symbol}")
        return None
    if isinstance(rates, (list, tuple)):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df
    return None


def backtest(df, params):
    ma_type = params.get("ma_type", "kama")
    ema_fast = calc_ma(df, params["ema_fast"], ma_type)
    ema_slow = calc_ma(df, params["ema_slow"], ma_type)
    atr = calc_atr_series(df, params["atr_period"])
    point = 0.01
    tick_value = 0.01

    trades = []
    position = None
    equity = [0.0]
    n = len(df)

    for i in range(max(params["ema_slow"], params["atr_period"]) + 50, n):
        bar = df.iloc[i]
        cur_atr = atr.iloc[i]
        if pd.isna(cur_atr) or cur_atr <= 0:
            continue

        cur_fast = ema_fast.iloc[i]
        cur_slow = ema_slow.iloc[i]
        prev_fast = ema_fast.iloc[i - 1]
        prev_slow = ema_slow.iloc[i - 1]

        buy_signal = prev_fast <= prev_slow and cur_fast > cur_slow
        sell_signal = prev_fast >= prev_slow and cur_fast < cur_slow

        if position is not None:
            is_long = position["type"] == "buy"
            hit_sl = (is_long and bar["low"] <= position["sl"]) or (not is_long and bar["high"] >= position["sl"])
            hit_tp = (is_long and bar["high"] >= position["tp"]) or (not is_long and bar["low"] <= position["tp"])

            if hit_sl or hit_tp:
                exit_price = position["sl"] if hit_sl else position["tp"]
                pnl_pips = (
                    (exit_price - position["entry"]) / point if is_long else (position["entry"] - exit_price) / point
                )
                pnl = pnl_pips * tick_value
                trades.append(
                    {
                        "entry_bar": position["entry_bar"],
                        "exit_bar": i,
                        "type": position["type"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "pnl": pnl,
                    }
                )
                equity.append(equity[-1] + pnl)
                position = None
                continue

            opp_signal = (is_long and sell_signal) or (not is_long and buy_signal)
            if opp_signal:
                exit_price = bar["close"]
                pnl_pips = (
                    (exit_price - position["entry"]) / point if is_long else (position["entry"] - exit_price) / point
                )
                pnl = pnl_pips * tick_value
                trades.append(
                    {
                        "entry_bar": position["entry_bar"],
                        "exit_bar": i,
                        "type": position["type"],
                        "entry": position["entry"],
                        "exit": exit_price,
                        "pnl": pnl,
                    }
                )
                equity.append(equity[-1] + pnl)
                position = None

        if buy_signal and position is None:
            sl = bar["close"] - cur_atr / params["atr_sl_mult"]
            tp = bar["close"] + cur_atr / params["atr_sl_mult"] * params["rr"]
            position = {"type": "buy", "entry": bar["close"], "sl": sl, "tp": tp, "entry_bar": i}
        elif sell_signal and position is None:
            sl = bar["close"] + cur_atr / params["atr_sl_mult"]
            tp = bar["close"] - cur_atr / params["atr_sl_mult"] * params["rr"]
            position = {"type": "sell", "entry": bar["close"], "sl": sl, "tp": tp, "entry_bar": i}

    if position is not None:
        pnl_pips = (
            (df.iloc[-1]["close"] - position["entry"]) / point
            if position["type"] == "buy"
            else (position["entry"] - df.iloc[-1]["close"]) / point
        )
        pnl = pnl_pips * tick_value
        trades.append(
            {
                "entry_bar": position["entry_bar"],
                "exit_bar": n - 1,
                "type": position["type"],
                "entry": position["entry"],
                "exit": df.iloc[-1]["close"],
                "pnl": pnl,
            }
        )
        equity.append(equity[-1] + pnl)

    return trades, equity


def compute_metrics(trades, equity):
    if not trades:
        return {"trades": 0, "return_pct": 0, "max_dd": 0, "sharpe": 0, "win_rate": 0}

    bars_per_day = 24  # H1 timeframe
    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    equity_arr = np.array(equity)
    running_max = np.maximum.accumulate(equity_arr)
    dd = running_max - equity_arr
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float("inf")

    period = len(equity) / (365.25 * bars_per_day)
    ann_return = total_pnl / period if period > 0 else 0
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1
    sharpe = (ann_return / std_pnl) * np.sqrt(365.25 * bars_per_day) if std_pnl > 0 else 0
    calmar = abs(total_pnl / max_dd) if max_dd > 0 else 0

    return {
        "trades": len(trades),
        "return_pct": total_pnl,
        "max_dd_pct": max_dd,
        "sharpe": sharpe,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor if profit_factor != float("inf") else 999.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def walk_forward(df, params):
    n = len(df)
    train_years = 3
    test_months = 9
    bars_per_day = 24  # H1 timeframe
    train_bars = train_years * 365 * bars_per_day
    test_bars = test_months * 30 * bars_per_day

    folds = []
    start = max(2000, int(n * 0.2))
    while start + train_bars + test_bars < n:
        folds.append((start, start + train_bars, start + train_bars + test_bars))
        start += test_bars

    if not folds:
        train_end = int(n * 0.7)
        folds = [(int(n * 0.1), train_end, n)]

    oos_trades = []
    oos_equity = [0.0]

    for train_start, train_end, test_end in folds:
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        _, train_metrics = backtest(train_df, params)
        best_oos_params = params

        test_trades, test_equity = backtest(test_df, best_oos_params)

        len(oos_equity) - 1
        for i in range(1, len(test_equity)):
            oos_equity.append(oos_equity[-1] + test_equity[i] - test_equity[i - 1])

        for t in test_trades:
            t["entry_bar"] += train_end
            t["exit_bar"] += train_end
            oos_trades.append(t)

    return oos_trades, oos_equity


def main():
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")
    symbol = settings.get("PORTFOLIO", "symbols", fallback=settings["TRADING"]["symbol"]).split(",")[0].strip()
    tf_name = settings["TRADING"]["timeframe"]
    tf = getattr(mt5, f"TIMEFRAME_{tf_name}", mt5.TIMEFRAME_H1)

    if not init_mt5():
        return

    mt5.symbol_select(symbol, True)
    logging.info(f"Fetching {BAR_COUNT} bars of {symbol} {tf_name}...")
    df = fetch_data(symbol, tf)
    if df is None:
        mt5.shutdown()
        return

    logging.info(f"Loaded {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    mt5.shutdown()

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    logging.info(f"Testing {len(combos)} parameter combinations...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        params["atr_period"] = 14

        trades, equity = walk_forward(df, params)
        metrics = compute_metrics(trades, equity)
        if metrics["trades"] < 10:
            continue
        metrics["params"] = params
        results.append(metrics)

    results.sort(key=lambda r: r["calmar"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"WALK-FORWARD OPTIMIZATION RESULTS — {symbol} {tf_name}")
    print(f"{'=' * 80}")
    print(
        f"{'Rank':<5} {'Params':<50} {'Trades':<7} {'Return':<10} "
        f"{'MaxDD':<10} {'Sharpe':<8} {'Calmar':<8} {'Win%':<6} {'PF':<6}"
    )
    print(f"{'-' * 80}")

    for i, r in enumerate(results[:15]):
        ps = f"EMA{r['params']['ema_fast']}/{r['params']['ema_slow']} "
        ps += f"SL={r['params']['atr_sl_mult']} "
        ps += f"RR={r['params']['rr']} "
        ps += f"ADX={r['params']['adx_trend_threshold']}"
        print(
            f"{i + 1:<5} {ps:<50} {r['trades']:<7} {r['return_pct']:<+10.2f} "
            f"{r['max_dd_pct']:<10.2f} {r['sharpe']:<8.2f} {r['calmar']:<8.2f} "
            f"{r['win_rate'] * 100:<5.1f}% {r['profit_factor']:<6.2f}"
        )

    best = results[0] if results else None
    if best:
        print(f"\n{'=' * 80}")
        print(f"BEST PARAMETERS (Calmar={best['calmar']:.2f}, Sharpe={best['sharpe']:.2f})")
        print(f"{'=' * 80}")
        print(f"  EMA Fast:         {best['params']['ema_fast']}")
        print(f"  EMA Slow:         {best['params']['ema_slow']}")
        print(f"  ATR SL Mult:      {best['params']['atr_sl_mult']}")
        print(f"  Risk/Reward:      {best['params']['rr']}")
        print(f"  ADX Threshold:    {best['params']['adx_trend_threshold']}")
        print(f"  OOS Trades:       {best['trades']}")
        print(f"  Win Rate:         {best['win_rate'] * 100:.1f}%")
        print(f"  Profit Factor:    {best['profit_factor']:.2f}")
        print(f"  Return (OOS):     ${best['return_pct']:.2f}")
        print(f"  Max DD (OOS):     ${best['max_dd_pct']:.2f}")
        print(f"  Calmar Ratio:     {best['calmar']:.2f}")
        print(f"  Sharpe Ratio:     {best['sharpe']:.2f}")

    if best:
        print(f"\n{'=' * 80}")
        print("Running Monte Carlo robustness validation (5000 simulations)...")
        print(f"{'=' * 80}")
        mc_trades, mc_equity = walk_forward(df, best["params"])
        mc_pnls = np.array([t["pnl"] for t in mc_trades])
        if len(mc_pnls) > 10:
            mc_report = mc_validation.compute_mc_report(mc_pnls, best, n_simulations=5000)
            mc_validation.print_mc_report(mc_report)
        else:
            print("  ⚠ Too few trades for MC validation — skipping\n")

    out_path = LOG_DIR / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out_path, "w") as f:
        f.write(
            "rank,ema_fast,ema_slow,atr_sl_mult,rr,adx_threshold,trades,return,max_dd,sharpe,calmar,win_rate,profit_factor\n"
        )
        for i, r in enumerate(results):
            p = r["params"]
            f.write(
                f"{i + 1},{p['ema_fast']},{p['ema_slow']},{p['atr_sl_mult']},{p['rr']},"
                f"{p['adx_trend_threshold']},{r['trades']},{r['return_pct']:.2f},"
                f"{r['max_dd_pct']:.2f},{r['sharpe']:.2f},{r['calmar']:.2f},"
                f"{r['win_rate']:.4f},{r['profit_factor']:.4f}\n"
            )
    logging.info(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
