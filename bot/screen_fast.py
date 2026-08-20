"""Quick symbol screen — one profile per symbol."""

import csv
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

logging.disable(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from datetime import datetime, timedelta  # noqa: E402

import pandas as pd  # noqa: E402
from _mt5 import mt5  # noqa: E402
from backtest import Backtest  # noqa: E402
from credentials import load_credentials  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"

PROFILES = {
    "FOREX": {"ema_fast": 10, "ema_slow": 40, "atr_sl_mult": 1.5, "rr": 2.0, "adx": 25, "ma_type": "kama"},
    "INDICES": {"ema_fast": 15, "ema_slow": 60, "atr_sl_mult": 1.5, "rr": 2.0, "adx": 25, "ma_type": "kama"},
    "CRYPTO": {"ema_fast": 8, "ema_slow": 24, "atr_sl_mult": 1.0, "rr": 3.0, "adx": 25, "ma_type": "vidya"},
    "COMMOD": {"ema_fast": 10, "ema_slow": 40, "atr_sl_mult": 1.5, "rr": 2.5, "adx": 25, "ma_type": "kama"},
    "STOCKS": {"ema_fast": 5, "ema_slow": 20, "atr_sl_mult": 1.0, "rr": 2.0, "adx": 25, "ma_type": "kama"},
}


def classify(sym):
    s = sym.upper()
    if any(
        x in s
        for x in [
            "USD",
            "JPY",
            "GBP",
            "CHF",
            "AUD",
            "NZD",
            "CAD",
            "NOK",
            "SEK",
            "DKK",
            "PLN",
            "SGD",
            "HKD",
            "ZAR",
            "MXN",
            "TRY",
        ]
    ):
        cats = {
            "JPY": "FOREX",
            "USD": "FOREX",
            "GBP": "FOREX",
            "CHF": "FOREX",
            "AUD": "FOREX",
            "NZD": "FOREX",
            "CAD": "FOREX",
        }
        for k, v in cats.items():
            if k in s and k not in ("USD",) or (k == "USD" and s.endswith("USD.raw")):
                return v
        return "FOREX"
    if any(
        x in s
        for x in ["US500", "US30", "UK100", "DE40", "AUS200", "CN50", "FR40", "JP225", "ES35", "STOXX", "NAS100", "SPX"]
    ):
        return "INDICES"
    if any(
        x in s
        for x in [
            "BTC",
            "ETH",
            "DOG",
            "LTC",
            "XRP",
            "ADA",
            "SOL",
            "DOT",
            "BNB",
            "AVAX",
            "MATIC",
            "ATOM",
            "LINK",
            "UNI",
            "FIL",
            "ICP",
            "NEAR",
            "APT",
            "ARB",
            "OP",
            "SUI",
            "SEI",
            "TIA",
            "INJ",
            "RUNE",
            "AAVE",
            "CRV",
            "MKR",
            "COMP",
            "YFI",
            "SNX",
            "SUSHI",
            "CAKE",
            "AXS",
            "SAND",
            "MANA",
            "ENJ",
            "GALA",
            "CHZ",
            "BCH",
            "EOS",
            "TRX",
            "XLM",
            "ALGO",
            "FTM",
            "AVX",
            "AVGO",
        ]
    ):
        return "CRYPTO"
    if any(x in s for x in ["XAU", "XAG", "XPT", "XPD", "UKO", "USO", "XNG", "XAUEUR"]):
        return "COMMOD"
    return "STOCKS"


def run():
    creds = load_credentials()
    if not mt5.initialize(
        login=creds["account"],
        password=creds["password"],
        server=creds["server"],
    ):
        print("MT5 init failed")
        return

    raw = sorted([s.name for s in mt5.symbols_get() if ".raw" in s.name])
    print(f"Symbols: {len(raw)}")

    end = datetime.now()
    start = end - timedelta(days=730)
    tf = mt5.TIMEFRAME_H1
    results = []

    for sym in raw:
        mt5.symbol_select(sym, True)
        sinfo = mt5.symbol_info(sym)
        if not sinfo:
            continue
        rates = mt5.copy_rates_range(sym, tf, start, end)
        if rates is None or len(rates) < 200:
            continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        cat = classify(sym)
        prof = PROFILES[cat]
        point = sinfo.point or 0.01
        tv = sinfo.trade_tick_value or 0.01
        vs = sinfo.volume_step if sinfo.volume_step > 0 else 0.01

        try:
            p = {
                "symbol": sym,
                "timeframe": tf,
                "ema_fast": prof["ema_fast"],
                "ema_slow": prof["ema_slow"],
                "ma_type": prof["ma_type"],
                "atr_sl_mult": prof["atr_sl_mult"],
                "rr": prof["rr"],
                "adx_trend_threshold": prof["adx"],
                "adx_trend_period": 14,
                "risk_percent": 1.0,
                "initial_balance": 500000.0,
                "max_positions_per_symbol": 1,
                "max_risk_ratio": 2.0,
                "point": point,
                "tick_value": tv,
                "volume_step": vs,
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
            bt = Backtest(df, p)
            r = bt.run()
            pf = r.get("profit_factor", 0) if r else 0
            wr = r.get("win_rate", 0) if r else 0
            ret = r.get("total_return", 0) if r else 0
            dd = r.get("max_dd", 0) if r else 0
            n = r.get("n_trades", 0) if r else 0
        except Exception:
            pf = wr = ret = dd = n = 0

        results.append(
            {
                "symbol": sym,
                "cat": cat,
                "pf": round(pf, 4),
                "wr": round(wr, 4),
                "ret": round(ret, 2),
                "dd": round(dd, 2),
                "trades": n,
                "bars": len(df),
            }
        )
        print(f"  {sym:25s} {cat:8s} PF={pf:<8.4f} WR={wr:<6.2%} Ret={ret:>8.0f} Trades={n}")

    mt5.shutdown()
    results.sort(key=lambda r: r["pf"], reverse=True)

    out = LOG_DIR / "symbol_screen.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "cat", "pf", "wr", "ret", "dd", "trades", "bars"])
        w.writeheader()
        w.writerows(results)

    print(f"\nSaved to {out}")
    print("\nTop 15 by PF:")
    for r in results[:15]:
        print(
            f"  {r['symbol']:25s} {r['cat']:8s} PF={r['pf']:<8.4f} WR={r['wr']:<6.2%} "
            f"Ret={r['ret']:>8.0f} DD={r['dd']:<8.0f} Trades={r['trades']}"
        )


if __name__ == "__main__":
    run()
