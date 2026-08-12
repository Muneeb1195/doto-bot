import logging
import time
from datetime import datetime

import requests

COLORS = {
    "BUY": 0x00FF00,
    "SELL": 0xFF0000,
    "OPEN": 0x5865F2,
    "CLOSE": 0x808080,
    "PARTIAL": 0x00BFFF,
    "REVERSAL": 0xFFA500,
    "MR": 0x9B59B6,
    "ERROR": 0xE74C3C,
    "INFO": 0x3498DB,
    "DAILY": 0x1ABC9C,
}


def send_alert(discord_url, event_type, title, fields, color=None):
    if not discord_url or discord_url == "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE":
        return
    if color is None:
        color = COLORS.get(event_type, COLORS["INFO"])
    embed = {
        "title": title,
        "color": color,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fields": [{"name": k, "value": str(v), "inline": True} for k, v in fields],
    }
    for attempt in range(2):
        try:
            resp = requests.post(
                discord_url,
                json={"embeds": [embed]},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                return
            if resp.status_code == 429:
                logging.warning("Discord rate limited, retrying...")
                time.sleep(1)
                continue
            logging.error(f"Discord send failed: {resp.status_code} {resp.text[:200]}")
            return
        except Exception as e:
            logging.warning(f"Discord send error: {e}")
            return


def trade_open(discord_url, symbol, side, volume, entry, sl, tp, atr, regime):
    send_alert(
        discord_url,
        side,
        f"📈 {side.upper()} {symbol}",
        [
            ("Volume", f"{volume:.2f}"),
            ("Entry", f"{entry:.2f}"),
            ("SL", f"{sl:.2f}"),
            ("TP", f"{tp:.2f}"),
            ("ATR", f"{atr:.2f}"),
            ("Regime", regime),
        ],
        COLORS[side.upper()],
    )


def trade_close(discord_url, symbol, side, volume, entry, exit_p, pnl, pips, reason):
    emoji = "✅" if pnl > 0 else "❌"
    send_alert(
        discord_url,
        "CLOSE",
        f"{emoji} CLOSE {symbol} {side.upper()}",
        [
            ("Volume", f"{volume:.2f}"),
            ("Entry", f"{entry:.2f}"),
            ("Exit", f"{exit_p:.2f}"),
            ("P&L", f"Rs.{pnl:+.2f}" if abs(pnl) < 1000 else f"Rs.{pnl:+,.2f}"),
            ("Pips", f"{pips:.1f}"),
            ("Reason", reason),
        ],
        COLORS["CLOSE"] if pnl <= 0 else COLORS[side.upper()],
    )


def trade_partial(discord_url, symbol, side, volume, price, pnl):
    send_alert(
        discord_url,
        "PARTIAL",
        f"🔵 PARTIAL {symbol} {side.upper()}",
        [
            ("Closed", f"{volume:.2f}"),
            ("Price", f"{price:.2f}"),
            ("P&L", f"Rs.{pnl:+.2f}"),
        ],
    )


def daily_summary(discord_url, balance, equity, daily_pnl, positions, win_rate, trades_today):
    send_alert(
        discord_url,
        "DAILY",
        f"📊 Daily Summary — {datetime.now().strftime('%Y-%m-%d')}",
        [
            ("Balance", f"Rs.{balance:,.2f}"),
            ("Equity", f"Rs.{equity:,.2f}"),
            ("Daily P&L", f"Rs.{daily_pnl:+,.2f}"),
            ("Open Positions", str(positions)),
            ("Win Rate", f"{win_rate:.1f}%" if win_rate else "N/A"),
            ("Trades Today", str(trades_today)),
        ],
    )


def bot_start(discord_url, symbols, balance):
    send_alert(
        discord_url,
        "INFO",
        "🚀 Bot Started",
        [
            ("Symbols", ", ".join(symbols)),
            ("Balance", f"Rs.{balance:,.2f}"),
            ("Time", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ],
    )
