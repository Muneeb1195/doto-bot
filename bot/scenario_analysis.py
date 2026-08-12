#!/usr/bin/env python3
"""Scenario analysis / stress testing for Doto MT5 portfolio."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "bot"))
from mt5_connect import login_account, mt5_call  # noqa: E402

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

REPORT_FILE = LOGS_DIR / "scenario_report.json"

SCENARIOS = {
    "market_crash": {
        "label": "Market Crash",
        "shocks": {"US30.raw": -8.0, "BTCUSD.raw": -15.0, "XAU500.raw": 2.0, "GBPJPY.raw": -5.0, "NZDUSD.raw": -3.0},
    },
    "gold_spike": {
        "label": "Gold Spike",
        "shocks": {"XAU500.raw": 6.0, "NZDUSD.raw": -2.0},
    },
    "crypto_crash": {
        "label": "Crypto Crash",
        "shocks": {"BTCUSD.raw": -20.0, "XAU500.raw": 4.0, "US30.raw": 1.0},
    },
    "usd_rally": {
        "label": "USD Rally",
        "shocks": {"NZDUSD.raw": -3.0, "GBPJPY.raw": -3.0, "XAU500.raw": -3.0, "US30.raw": -1.0, "BTCUSD.raw": 1.0},
    },
    "stagflation": {
        "label": "Stagflation",
        "shocks": {"US30.raw": -5.0, "XAU500.raw": 3.0, "BTCUSD.raw": -5.0, "GBPJPY.raw": -4.0, "NZDUSD.raw": -2.0},
    },
    "risk_on": {
        "label": "Risk-On Rally",
        "shocks": {"US30.raw": 5.0, "XAU500.raw": -2.0, "BTCUSD.raw": 8.0, "GBPJPY.raw": 3.0, "NZDUSD.raw": 2.0},
    },
}


def get_position_pkr(pos, sinfo):
    if sinfo is None or sinfo.point == 0 or sinfo.trade_tick_value == 0:
        return 0.0
    point = sinfo.point
    tick_value = sinfo.trade_tick_value
    is_long = pos.type == mt5.ORDER_TYPE_BUY
    current_price = pos.price_current
    entry = pos.price_open
    volume = pos.volume
    if is_long:
        return (current_price - entry) / point * tick_value * volume
    else:
        return (entry - current_price) / point * tick_value * volume


def compute_scenario_pnl(pos, sinfo, shock_pct):
    if sinfo is None or sinfo.point == 0 or sinfo.trade_tick_value == 0:
        return 0.0
    point = sinfo.point
    tick_value = sinfo.trade_tick_value
    is_long = pos.type == mt5.ORDER_TYPE_BUY
    entry = pos.price_open
    volume = pos.volume
    # shock_pct is a MARKET price move on the symbol — it applies to the price
    # the same way regardless of position direction. Previously the sign was
    # flipped for shorts, which made shorts show a LOSS during a crash (they
    # should profit) and inverted every stress scenario for short books
    # (agent audit H2).
    shocked_price = entry * (1 + shock_pct / 100)
    if is_long:
        return (shocked_price - entry) / point * tick_value * volume
    else:
        return (entry - shocked_price) / point * tick_value * volume


def run_scenario_analysis():
    if not login_account():
        print("MT5 init/login failed — see log for details")
        return

    account_info = mt5_call(mt5.account_info, _timeout=5)
    if account_info is None:
        print("Cannot get account info")
        mt5_call(mt5.shutdown, _timeout=5)
        return

    balance = account_info.balance
    equity = account_info.equity
    cb_pct = 15.0
    positions = mt5_call(mt5.positions_get, _timeout=5)
    if positions is None:
        positions = []

    sinfo_cache = {}
    for pos in positions:
        if pos.symbol not in sinfo_cache:
            mt5_call(mt5.symbol_select, pos.symbol, True, _timeout=10)
            sinfo_cache[pos.symbol] = mt5_call(mt5.symbol_info, pos.symbol, _timeout=5)

    print(f"\n{'=' * 70}")
    print("  SCENARIO ANALYSIS REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S PKT')}")
    print(f"{'=' * 70}")
    print(f"  Balance: Rs.{balance:,.0f}  |  Equity: Rs.{equity:,.0f}  |  Open Positions: {len(positions)}")
    print(f"{'=' * 70}")

    if not positions:
        print("\n  No open positions — scenario impact is Rs.0.00 for all scenarios.")
        print("\n  Historical worst-case overlay requires backtest data (not available).")
        results = {}
        current_pnl_total = 0.0
        for sc_key, sc in SCENARIOS.items():
            results[sc_key] = {"label": sc["label"], "pnl_pkr": 0.0, "pnl_pct": 0.0}
    else:
        print(f"\n  {'Position':<25} {'Type':<6} {'Volume':<8} {'Entry':<12} {'Current PnL':<14}")
        print(f"  {'-' * 65}")
        current_pnl_total = 0.0
        for pos in positions:
            sinfo = sinfo_cache.get(pos.symbol)
            pnl = get_position_pkr(pos, sinfo)
            pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            current_pnl_total += pnl
            print(f"  {pos.symbol:<25} {pos_type:<6} {pos.volume:<8.2f} {pos.price_open:<12.5f} Rs.{pnl:<+10.2f}")
        print(f"  {'-' * 65}")
        print(f"  {'Total Current PnL':<55} Rs.{current_pnl_total:<+10.2f}")

        print(f"\n  {'SCENARIO':<20} {'PnL (PKR)':<16} {'% of Balance':<16} {'Status':<16}")
        print(f"  {'-' * 68}")

        results = {}
        for sc_key, sc in SCENARIOS.items():
            scenario_pnl = 0.0
            for pos in positions:
                sinfo = sinfo_cache.get(pos.symbol)
                shock = sc["shocks"].get(pos.symbol)
                if shock is None:
                    continue
                pnl_impact = compute_scenario_pnl(pos, sinfo, shock)
                scenario_pnl += pnl_impact

            pnl_pct = (scenario_pnl / balance * 100) if balance > 0 else 0.0
            status = "OK" if abs(pnl_pct) < cb_pct else "CIRCUIT BREAKER"
            results[sc_key] = {
                "label": sc["label"],
                "pnl_pkr": round(scenario_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "status": status,
            }
            print(f"  {sc['label']:<20} Rs.{scenario_pnl:<+12.2f} {pnl_pct:<+15.2f}% {status:<16}")

    max_loss = min((r["pnl_pct"] for r in results.values()), default=0.0)
    max_loss_scenario = next((k for k, v in results.items() if v["pnl_pct"] == max_loss), "none")
    print(f"\n  {'=' * 68}")
    print(f"  Worst-case scenario: {results.get(max_loss_scenario, {}).get('label', 'N/A')} ({max_loss:.1f}%)")
    print(f"  Circuit breaker threshold: {cb_pct:.0f}% DD")
    if abs(max_loss) >= cb_pct:
        print(f"  ⚠  CIRCUIT BREAKER RISK: {abs(max_loss):.1f}% loss exceeds {cb_pct:.0f}% limit")
    else:
        margin = cb_pct - abs(max_loss)
        print(f"  Headroom to circuit breaker: {margin:.1f}%")
    print(f"{'=' * 68}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "balance": balance,
        "equity": equity,
        "open_positions": len(positions),
        "current_pnl_total": current_pnl_total if positions else 0,
        "results": results,
        "circuit_breaker_pct": cb_pct,
        "max_loss_pct": max_loss,
        "max_loss_scenario": max_loss_scenario,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to {REPORT_FILE}")

    mt5_call(mt5.shutdown, _timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario analysis for Doto MT5 portfolio")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()
    run_scenario_analysis()
    if args.json:
        with open(REPORT_FILE) as f:
            print(f.read())
