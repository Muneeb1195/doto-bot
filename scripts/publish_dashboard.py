#!/usr/bin/env python3
"""Publish a sanitized public snapshot of the bot dashboard to GitHub Pages.

Reads the live bot artifacts (data/dashboard_state.json, logs/trades.csv,
logs/bot.log) and produces a static site — index.html + state.json +
trades.json + bot.log — that is pushed to the ``gh-pages`` branch of a
PUBLIC repository so it can be served by GitHub Pages.

SENSITIVE FIELDS ARE STRIPPED. The private two-tier dashboard (FastAPI on
homer, reached via Tailscale) remains the only place where balance, equity,
positions, margin, and per-trade PnL are visible. This script intentionally
does not copy those values to the public site.

Designed to run on the homer server via a systemd user timer (every 5 min).

Usage:
    python scripts/publish_dashboard.py [--force]
    python scripts/publish_dashboard.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_DIR / "data" / "dashboard_state.json"
TRADES_CSV = REPO_DIR / "logs" / "trades.csv"
BOT_LOG = REPO_DIR / "logs" / "bot.log"

# Public gh-pages repo. Change via --public-repo if you fork.
PUBLIC_REPO = os.environ.get("DOTO_PUBLIC_REPO", "Muneeb1195/doto-dashboard")
PUBLIC_BRANCH = "gh-pages"

LOG_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[(INFO|WARNING|ERROR)\]")
MAX_LOG_LINES = 60
MAX_LOG_BYTES = 262144

CLOSED_EVENTS = {"CLOSE", "REVERSAL", "MR_EXIT", "CHANDELIER", "MANUAL_CLOSE"}
PARTIAL_EVENTS = {"SCALE_OUT", "PARTIAL"}

STATIC_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Doto Bot — Public Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root { --bg:#0f0f1a; --card:#1a1a2e; --line:#2a2a45; --txt:#e8e8f0; --dim:#8888bb;
        --green:#00d4aa; --red:#ff4466; --blue:#4488ff; }
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--txt); font:14px/1.5 system-ui,sans-serif; padding:20px; }
h1 { font-size:20px; margin-bottom:4px; }
.sub { color:var(--dim); margin-bottom:18px; font-size:13px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-bottom:18px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card h3 { font-size:12px; text-transform:uppercase; letter-spacing:.5px; color:var(--dim); margin-bottom:10px; }
.metric { font-size:20px; font-weight:600; }
.metric.small { font-size:16px; }
.pos { color:var(--green); } .neg { color:var(--red); } .warn { color:#ffcc55; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }
th { color:var(--dim); font-weight:600; font-size:12px; text-transform:uppercase; }
section { margin-bottom:22px; }
pre { background:#111122; border:1px solid var(--line); border-radius:8px; padding:12px; font-size:12px;
       white-space:pre-wrap; word-break:break-word; max-height:380px; overflow-y:auto; }
.badge { display:inline-block; padding:2px 8px; border-radius:20px; font-size:12px; font-weight:600; }
.badge.ok { background:#14352c; color:var(--green); } .badge.bad { background:#3a1620; color:var(--red); }
canvas { max-height:220px; }
</style>
</head>
<body>
<h1>Doto MT5 Bot — Public Dashboard</h1>
<div class="sub">Sanitized snapshot — balances, positions, and per-trade P&L are private. Updated every 5 minutes.</div>

<div class="grid">
  <div class="card"><h3>Total Trades</h3><div class="metric" id="m_trades">&ndash;</div></div>
  <div class="card"><h3>Win Rate</h3><div class="metric" id="m_win">&ndash;</div></div>
  <div class="card"><h3>Profit Factor</h3><div class="metric" id="m_pf">&ndash;</div></div>
  <div class="card"><h3>Regime</h3><div class="metric small" id="m_regime">&ndash;</div></div>
  <div class="card"><h3>MT5 Status</h3><div id="m_health">&ndash;</div></div>
  <div class="card"><h3>Last Update</h3><div class="metric small" id="m_updated">&ndash;</div></div>
</div>

<section class="card" id="sec_breakdown">
  <h3>Signal Filter Breakdown</h3>
  <div style="display:flex; gap:24px; flex-wrap:wrap;">
    <table id="tbl_filters"><thead><tr><th>Symbol</th><th>Signals</th><th>HTF Trend</th>
      <th>Regime Gate</th><th>ML Gate</th><th>Sanity</th><th>No Signal</th><th>Tail Risk</th>
      </tr></thead><tbody></tbody></table>
  </div>
</section>

<section class="card">
  <h3>Regime Distribution</h3>
  <div style="display:flex; gap:24px; flex-wrap:wrap;">
    <canvas id="chart_regime" style="max-width:340px; max-height:180px;"></canvas>
    <table id="tbl_regime"><thead><tr><th>Regime</th><th>Symbols</th></tr></thead><tbody></tbody></table>
  </div>
</section>

<section class="card">
  <h3>Performance by Symbol</h3>
  <table id="tbl_perf"><thead><tr><th>Symbol</th><th>Trades</th><th>Win Rate</th>
    <th>Avg Pips</th></tr></thead><tbody></tbody></table>
</section>

<section class="card">
  <h3>Bot Log (last 60 lines)</h3>
  <pre id="botlog">Loading&hellip;</pre>
</section>

<script>
const CLOSED = ["CLOSE","REVERSAL","MR_EXIT","CHANDELIER","MANUAL_CLOSE"];
let state = null, trades = null;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function pct(x) { return (x === null || x === undefined) ? "&ndash;" : (x * 100).toFixed(1) + "%"; }

async function load() {
  try { const [r1, r2] = await Promise.all([fetch("state.json"), fetch("trades.json")]);
        state = await r1.json(); trades = await r2.json(); } catch (e) { return; }

  document.getElementById("m_trades").textContent = trades.total_trades;
  document.getElementById("m_win").textContent = pct(trades.win_rate);
  document.getElementById("m_pf").textContent =
    trades.profit_factor === null ? "&ndash;" : trades.profit_factor.toFixed(2);

  const regimeCounts = state.regime_counts || {};
  const dominant = Object.entries(regimeCounts).sort((a,b) => b[1]-a[1])[0];
  document.getElementById("m_regime").textContent = dominant ? dominant[0] : "n/a";

  const h = state.health || {};
  const connected = h.connected;
  document.getElementById("m_health").innerHTML =
    '<span class="badge ' + (connected ? "ok" : "bad") + '">' + (connected ? "Connected" : "Disconnected") +
    '</span> <span style="color:var(--dim)">' + esc(h.server || "") + '</span>';

  document.getElementById("m_updated").textContent = state.updated || "n/a";

  renderFilters(state.filters);
  renderRegimes(regimeCounts);
  renderPerf(trades.by_symbol || {});
}

function renderFilters(filters) {
  const tb = document.querySelector("#tbl_filters tbody");
  tb.innerHTML = "";
  if (!filters) return;
  const keys = ["signals","htf_trend","regime_gate","ml_gate","sanity","no_signal","tail_risk"];
  for (const [sym, f] of Object.entries(filters)) {
    const tr = document.createElement("tr");
    const cells = [sym];
    for (const k of keys) cells.push(f && f[k] !== undefined ? f[k] : 0);
    for (const c of cells) { const td = document.createElement("td"); td.textContent = c; tr.appendChild(td); }
    tb.appendChild(tr);
  }
}

function renderRegimes(regimeCounts) {
  const tb = document.querySelector("#tbl_regime tbody");
  tb.innerHTML = "";
  const labels = [], data = [];
  for (const [k, v] of Object.entries(regimeCounts)) {
    const tr = document.createElement("tr");
    const td1 = document.createElement("td"); td1.textContent = k;
    const td2 = document.createElement("td"); td2.textContent = v;
    tr.appendChild(td1); tr.appendChild(td2); tb.appendChild(tr);
    labels.push(k); data.push(v);
  }
  if (window._rc) window._rc.destroy();
  const ctx = document.getElementById("chart_regime");
  if (labels.length) {
    window._rc = new Chart(ctx, { type: "doughnut",
      data: { labels, datasets: [{ data, backgroundColor: ["#4488ff","#00d4aa","#ffcc55","#ff4466","#8888bb"] }] },
      options: { plugins: { legend: { position: "right", labels: { color: "#8888bb" } } } } });
  }
}

function renderPerf(bySymbol) {
  const tb = document.querySelector("#tbl_perf tbody");
  tb.innerHTML = "";
  for (const [sym, p] of Object.entries(bySymbol)) {
    const tr = document.createElement("tr");
    for (const c of [sym, p.trades, pct(p.win_rate), p.avg_pips === null ? "&ndash;" : p.avg_pips.toFixed(1)]) {
      const td = document.createElement("td"); td.innerHTML = c; tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
}

async function loadLog() {
  try { const r = await fetch("bot.log"); const txt = await r.text();
        document.getElementById("botlog").textContent = txt.trim() || "(empty)"; }
  catch (e) { document.getElementById("botlog").textContent = "(log unavailable)"; }
}

load();
loadLog();
setInterval(load, 60000);
setInterval(loadLog, 60000);
</script>
</body>
</html>
"""


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _sanitize_state(state: dict) -> dict:
    """Keep only public-safe fields. Balances/equity/positions/margin are dropped."""
    regimes = state.get("regimes") or {}
    counts: dict[str, int] = {}
    for r in regimes.values():
        counts[r] = counts.get(r, 0) + 1
    return {
        "updated": state.get("timestamp"),
        "health": {
            "connected": (state.get("health") or {}).get("connected"),
            "server": (state.get("health") or {}).get("server"),
        },
        "regime_counts": counts,
        "filters": state.get("filters") or {},
    }


def _read_trades() -> list[dict]:
    if not TRADES_CSV.exists():
        return []
    rows = []
    try:
        with open(TRADES_CSV, "r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except (OSError, csv.Error):
        return []
    return rows


def _aggregate_trades(rows: list[dict]) -> dict:
    closed = [r for r in rows if (r.get("event") or "").strip() in CLOSED_EVENTS]
    total = len(closed)
    wins = [r for r in closed if _f(r.get("pnl")) > 0]
    losses = [r for r in closed if _f(r.get("pnl")) <= 0]
    gross_win = sum(_f(r.get("pnl")) for r in wins)
    gross_loss = abs(sum(_f(r.get("pnl")) for r in losses))
    by_symbol: dict[str, dict] = {}
    for r in closed:
        sym = r.get("symbol") or "?"
        s = by_symbol.setdefault(sym, {"trades": 0, "wins": 0, "pips": 0.0, "count": 0})
        s["count"] += 1
        if _f(r.get("pnl")) > 0:
            s["wins"] += 1
        s["pips"] += _f(r.get("pips"))
    for s in by_symbol.values():
        s["trades"] = s["count"]
        s["win_rate"] = (s["wins"] / s["count"]) if s["count"] else None
        s["avg_pips"] = (s["pips"] / s["count"]) if s["count"] else None
        s.pop("count", None)
        s.pop("wins", None)
    return {
        "total_trades": total,
        "win_rate": (len(wins) / total) if total else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (None if gross_win == 0 else float("inf")),
        "by_symbol": by_symbol,
    }


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _tail_log() -> str:
    if not BOT_LOG.exists():
        return "(no log yet)"
    try:
        with open(BOT_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - MAX_LOG_BYTES))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return "(log unavailable)"
    lines = [ln for ln in data.splitlines() if LOG_LINE_RE.match(ln)]
    return "\n".join(lines[-MAX_LOG_LINES:])


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="render but do not push")
    ap.add_argument("--force", action="store_true", help="push even if nothing changed")
    ap.add_argument("--public-repo", default=PUBLIC_REPO, help="public repo owner/name for gh-pages")
    ap.add_argument("--work", default=REPO_DIR / ".dashboard_public", help="working checkout dir")
    args = ap.parse_args()

    state = _read_json(STATE_FILE)
    if state is None:
        print("no dashboard_state.json — nothing to publish")
        return 0
    sanitized = _sanitize_state(state)
    trades = _aggregate_trades(_read_trades())
    log_text = _tail_log()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if not (work / ".git").exists():
        if args.dry_run:
            print(f"[dry-run] would clone {args.public_repo} into {work}")
        else:
            print(f"cloning {args.public_repo} into {work}")
            subprocess.run(
                ["git", "clone", f"https://github.com/{args.public_repo}.git", "-b", PUBLIC_BRANCH, str(work)],
                check=True, capture_output=True, text=True,
            )

    (work / "state.json").write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    (work / "trades.json").write_text(json.dumps(trades, indent=2), encoding="utf-8")
    (work / "bot.log").write_text(log_text, encoding="utf-8")
    (work / "index.html").write_text(STATIC_TEMPLATE, encoding="utf-8")

    _git(["add", "-A"], work)
    changed = _git(["status", "--porcelain"], work).stdout.strip()
    if not changed and not args.force:
        print("no changes — skipping push")
        return 0
    if args.dry_run:
        print(f"[dry-run] would commit+push {len(changed.splitlines())} changed file(s)")
        return 0

    _git(["add", "-A"], work)
    r = _git(["commit", "-m", "dashboard snapshot update"], work)
    if r.returncode != 0 and "nothing to commit" not in r.stderr:
        print("commit failed:", r.stderr.strip())
        return 1
    r = _git(["push", "origin", PUBLIC_BRANCH], work)
    if r.returncode != 0:
        print("push failed:", r.stderr.strip())
        return 1
    print("published dashboard snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
