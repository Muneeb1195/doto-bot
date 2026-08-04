import csv
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
TRADE_CSV = BASE_DIR / "logs" / "trades.csv"
STATE_JSON = BASE_DIR / "data" / "dashboard_state.json"

PKT = timezone(timedelta(hours=5))


def week_start(dt):
    return dt - timedelta(days=dt.weekday())


def compute_weekly_stats():
    cutoff = week_start(datetime.now(PKT).replace(hour=0, minute=0, second=0, microsecond=0))

    if not TRADE_CSV.exists():
        return None

    trades = []
    with open(TRADE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("exit_time"):
                continue
            try:
                t = datetime.fromisoformat(row["exit_time"])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=PKT)
                if t < cutoff:
                    continue
                trades.append(row)
            except (ValueError, KeyError):
                continue

    if not trades:
        return {"msg": "No closed trades this week.", "trades": 0}

    total_pnl = sum(float(t["pnl"]) for t in trades if t.get("pnl"))
    wins = [t for t in trades if float(t.get("pnl", 0)) > 0]
    losses = [t for t in trades if float(t.get("pnl", 0)) <= 0]
    wr = len(wins) / len(trades) * 100 if trades else 0
    gross_win = sum(float(t["pnl"]) for t in wins)
    gross_loss = abs(sum(float(t["pnl"]) for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 0

    per_symbol = {}
    for t in trades:
        sym = t.get("symbol", "?")
        pnl = float(t.get("pnl", 0))
        per_symbol.setdefault(sym, {"trades": 0, "pnl": 0.0, "wins": 0})
        per_symbol[sym]["trades"] += 1
        per_symbol[sym]["pnl"] += pnl
        if pnl > 0:
            per_symbol[sym]["wins"] += 1

    summary = {
        "trades": len(trades),
        "wr": wr,
        "pf": pf,
        "total_pnl": total_pnl,
        "avg_win": gross_win / len(wins) if wins else 0,
        "avg_loss": gross_loss / len(losses) if losses else 0,
        "best_trade": max(trades, key=lambda t: float(t.get("pnl", 0))) if trades else None,
        "worst_trade": min(trades, key=lambda t: float(t.get("pnl", 0))) if trades else None,
        "per_symbol": per_symbol,
    }
    return summary


def load_balance():
    if STATE_JSON.exists():
        try:
            with open(STATE_JSON) as f:
                state = json.load(f)
            return state.get("balance", 0)
        except Exception:
            logging.debug("Could not load dashboard state for weekly summary", exc_info=True)
    return 0


def send_weekly_discord(webhook_url):
    stats = compute_weekly_stats()
    balance = load_balance()

    embed = {
        "title": "Weekly Trading Summary",
        "color": 0x5865F2,
        "timestamp": datetime.now(PKT).isoformat(),
        "footer": {"text": f"Balance: Rs.{balance:,.2f}"},
        "fields": [],
    }

    if stats is None:
        embed["description"] = "No trade data found."
        embed["color"] = 0x808080
    elif stats.get("msg"):
        embed["description"] = stats["msg"]
        embed["color"] = 0x808080
    else:
        embed["description"] = f"**Week of {week_start(datetime.now(PKT)).strftime('%b %d, %Y')}**"
        embed["fields"] = [
            {"name": "Trades", "value": str(stats["trades"]), "inline": True},
            {"name": "Win Rate", "value": f"{stats['wr']:.1f}%", "inline": True},
            {"name": "Profit Factor", "value": f"{stats['pf']:.2f}", "inline": True},
            {"name": "Total P&L", "value": f"Rs.{stats['total_pnl']:+,.2f}", "inline": True},
            {"name": "Avg Win", "value": f"Rs.{stats['avg_win']:+,.2f}", "inline": True},
            {"name": "Avg Loss", "value": f"Rs.{stats['avg_loss']:+,.2f}", "inline": True},
        ]

        sym_lines = []
        for sym, data in sorted(stats["per_symbol"].items()):
            sr = data["wins"] / data["trades"] * 100 if data["trades"] else 0
            sym_lines.append(f"{sym}: {data['trades']}T {sr:.0f}% WR Rs.{data['pnl']:+,.0f}")
        if sym_lines:
            embed["fields"].append(
                {
                    "name": "Per Symbol",
                    "value": "\n".join(sym_lines[:10]),
                    "inline": False,
                }
            )

        best = stats["best_trade"]
        worst = stats["worst_trade"]
        if best:
            embed["fields"].append(
                {
                    "name": "Best Trade",
                    "value": f"{best['symbol']} {best['type']} Rs.{float(best['pnl']):+,.2f}",
                    "inline": True,
                }
            )
        if worst:
            embed["fields"].append(
                {
                    "name": "Worst Trade",
                    "value": f"{worst['symbol']} {worst['type']} Rs.{float(worst['pnl']):+,.2f}",
                    "inline": True,
                }
            )

        if stats["total_pnl"] >= 0:
            embed["color"] = 0x00FF00
        else:
            embed["color"] = 0xFF0000

    payload = {"embeds": [embed]}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        print(f"Discord: {resp.status_code}")
    except Exception as e:
        print(f"Discord error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python weekly_summary.py <webhook_url>")
        sys.exit(1)
    send_weekly_discord(sys.argv[1])
