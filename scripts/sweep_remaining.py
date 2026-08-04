"""Sweep remaining symbols with fresh models.

Resumes from where the previous sweep left off (checks which symbols
already have results in the sweep CSV). Saves progress after each symbol
so a crash/power-outage doesn't lose everything.

Usage:
    python scripts/sweep_remaining.py
"""
import sys
import csv
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
OPTIMIZER = BASE / "bot" / "optimize_params.py"
PYTHON = BASE / ".venv" / "Scripts" / "python.exe"
SWEEP_LOG = LOG_DIR / "sweep_all_20260724.log"
SWEEP_CSV = LOG_DIR / "symbol_sweep_20260724.csv"

ALL_SYMBOLS = [
    "ADAUSD", "AVXUSD", "BTCUSD", "DOGUSD", "ETHUSD", "EURJPY", "EURUSD",
    "GBPJPY", "GBPUSD", "IWM", "LTCUSD", "NZDUSD", "SOLUSD", "SPY",
    "US30", "US500", "USDJPY", "XAGUSD", "XAU500", "XAUUSD",
    "XNGUSD", "XPTUSD", "XRPUSD",
]


def get_done_symbols():
    """Read already-completed symbols from sweep CSV."""
    done = set()
    if SWEEP_CSV.exists():
        with open(SWEEP_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row["symbol"])
    return done


def get_best_from_log(symbol):
    """Extract best result for a symbol from the sweep log."""
    # Read the log and find the section for this symbol
    if not SWEEP_LOG.exists():
        return None
    content = SWEEP_LOG.read_text(errors="replace")
    # Find all "BEST for {symbol}.raw:" sections
    marker = f"BEST for {symbol}.raw:"
    idx = content.find(marker)
    if idx == -1:
        return None
    # Extract the section until the next BEST or end
    next_idx = content.find("BEST for", idx + len(marker))
    if next_idx == -1:
        section = content[idx:]
    else:
        section = content[idx:next_idx]

    best = {"symbol": symbol}
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("EMA") and "SL=" in line:
            # e.g. "EMA10/40  SL=1.5 RR=2.0 ADX=28 score=0.60  PF=0.73 Ret=Rs.-13149.5 DD=Rs.16365.0 WR=41.2% n=102  WF=0 windows=4"
            parts = line.split()
            best["ema"] = parts[0]
            for p in parts:
                if p.startswith("SL="):
                    best["sl"] = float(p[3:])
                elif p.startswith("RR="):
                    best["rr"] = float(p[3:])
                elif p.startswith("ADX="):
                    best["adx"] = int(p[4:])
                elif p.startswith("score="):
                    best["wf_score"] = float(p[6:])
                elif p.startswith("PF="):
                    best["pf"] = float(p[3:])
                elif p.startswith("Ret=Rs."):
                    best["ret"] = float(p[7:])
                elif p.startswith("n="):
                    best["n_trades"] = int(p[2:])
    return best if "wf_score" in best else None


def main():
    done = get_done_symbols()
    remaining = [s for s in ALL_SYMBOLS if s not in done]
    print(f"Already done: {sorted(done)}")
    print(f"Remaining: {remaining}\n")

    results = []
    # Load existing results
    if SWEEP_CSV.exists():
        with open(SWEEP_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)

    for sym in remaining:
        try:
            result = subprocess.run(
                [str(PYTHON), str(OPTIMIZER), "--symbols", f"{sym}.raw", "--two-phase"],
                capture_output=True, text=True, timeout=300,
                cwd=str(BASE),
            )
            # Append to sweep log
            with open(SWEEP_LOG, "a") as f:
                f.write(f"\n=== {sym} ===\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write(result.stderr)

            best = get_best_from_log(sym)
            if best is None:
                # Check if it was "No trades generated"
                if "No trades generated" in result.stdout:
                    print(f"  {sym}: no trades (skipped)")
                else:
                    print(f"  {sym}: no valid result")
                continue

            wf = best.get("wf_score", 0.0)
            n = best.get("n_trades", 0)
            if n < 30 or not np.isfinite(wf) or wf <= 0:
                print(f"  {sym}: skipped (n={n}, wf={wf:.2f})")
                continue

            results.append(best)
            print(f"  {sym}: WF={wf:.3f} PF={best.get('pf',0):.2f} n={n} ret=Rs.{best.get('ret',0):.0f}")

            # Save progress after each symbol
            results.sort(key=lambda r: float(r.get("wf_score", 0)), reverse=True)
            with open(SWEEP_CSV, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["symbol", "wf_score", "pf", "n_trades", "ret", "ema", "sl", "rr", "adx"])
                w.writeheader()
                for r in results:
                    w.writerow(r)

        except subprocess.TimeoutExpired:
            print(f"  {sym}: TIMEOUT")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    results.sort(key=lambda r: float(r.get("wf_score", 0)), reverse=True)
    print(f"\n=== SWEEP COMPLETE ===")
    print(f"Results: {SWEEP_CSV}")
    print(f"\nTop 12 by WF score:")
    for r in results[:12]:
        print(f"  {r['symbol']:12s} WF={float(r.get('wf_score',0)):.3f} PF={float(r.get('pf',0)):.2f} n={r.get('n_trades','?'):>3s} ret=Rs.{float(r.get('ret',0)):.0f}")


if __name__ == "__main__":
    main()
