"""Sweep all symbols with fresh models to find the best performers.

Runs `optimize_params.py --two-phase` for every symbol that has a trained
model, then ranks them by walk-forward score (real engine). Output:
logs/symbol_sweep_YYYYMMDD.csv sorted by WF score.

Usage:
    python scripts/sweep_symbols.py
"""
import csv
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
LOG_DIR = BASE / "logs"
OPTIMIZER = BASE / "bot" / "optimize_params.py"
PYTHON = BASE / ".venv" / "Scripts" / "python.exe"


def parse_best(output):
    """Extract best result from optimizer stdout."""
    best = {}
    for line in output.splitlines():
        if "BEST for" in line:
            # e.g. "  BEST for XAU500.raw:"
            best["symbol"] = line.split("BEST for")[1].split(":")[0].strip()
        elif "WF=" in line and "PF=" in line:
            # e.g. "XAU500.raw  EMA10/40  SL=1.5 RR=2.0 ADX=28 score=0.60  PF=0.73
            # Ret=Rs.-13149.5 DD=Rs.16365.0 WR=41.2% n=102  WF=0 windows=4"
            try:
                parts = line.split()
                # Find PF=, Ret=, n=, WF=
                for p in parts:
                    if p.startswith("PF="):
                        best["pf"] = float(p[3:])
                    elif p.startswith("Ret=Rs."):
                        best["ret"] = float(p[7:])
                    elif p.startswith("n="):
                        best["n_trades"] = int(p[2:])
                    elif p.startswith("WF="):
                        best["wf_score"] = float(p[3:])
                # EMA/SL/RR/ADX
                ema_part = [p for p in parts if p.startswith("EMA")][0]
                best["ema"] = ema_part
                for p in parts:
                    if p.startswith("SL="):
                        best["sl"] = float(p[3:])
                    elif p.startswith("RR="):
                        best["rr"] = float(p[3:])
                    elif p.startswith("ADX="):
                        best["adx"] = int(p[4:])
            except (ValueError, IndexError):
                pass
    return best


def main():
    # All symbols with trained models
    symbols = []
    for f in (BASE / "models").glob("model_*_raw.pkl"):
        sym = f.name[len("model_"):-len("_raw.pkl")]
        symbols.append(sym)

    print(f"Sweeping {len(symbols)} symbols with fresh models...\n")

    results = []
    for sym in sorted(symbols):
        try:
            result = subprocess.run(
                [str(PYTHON), str(OPTIMIZER), "--symbols", sym, "--two-phase"],
                capture_output=True, text=True, timeout=300,
                cwd=str(BASE),
            )
            best = parse_best(result.stdout)
            if not best or "wf_score" not in best:
                print(f"  {sym}: no valid result")
                continue

            wf = best.get("wf_score", 0.0)
            n = best.get("n_trades", 0)
            if n < 30 or not np.isfinite(wf) or wf <= 0:
                print(f"  {sym}: skipped (n={n}, wf={wf:.2f})")
                continue

            best["symbol"] = sym
            results.append(best)
            print(f"  {sym}: WF={wf:.3f} PF={best.get('pf',0):.2f} n={n} ret=Rs.{best.get('ret',0):.0f}")
        except subprocess.TimeoutExpired:
            print(f"  {sym}: TIMEOUT")
        except Exception as e:
            print(f"  {sym}: ERROR {e}")

    results.sort(key=lambda r: r["wf_score"], reverse=True)

    out = LOG_DIR / f"symbol_sweep_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "wf_score", "pf", "n_trades", "ret", "ema", "sl", "rr", "adx"])
        w.writeheader()
        for r in results:
            w.writerow(r)

    print("\n=== SWEEP COMPLETE ===")
    print(f"Results written to {out}")
    print("\nTop 12 by WF score:")
    for r in results[:12]:
        print(
            f"  {r['symbol']:12s} WF={r['wf_score']:.3f} PF={r.get('pf', 0):.2f} "
            f"n={r['n_trades']:3d} ret=Rs.{r.get('ret', 0):.0f}"
        )


if __name__ == "__main__":
    main()
