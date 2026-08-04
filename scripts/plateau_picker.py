"""Post-process optimizer CSV: pick the middle EMA pair from the top score cluster.

Usage:
    python scripts/plateau_picker.py logs/optimize_XAUUSD_raw.csv

Outputs the recommended param set (middle of plateau) to stdout as INI-ready lines.
Use --ini to format as settings.ini [STRATEGY:*] section.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def pick_middle_of_plateau(df, score_col="wf_score", pct=0.90, min_trades=3):
    """Select the middle EMA pair from the top WF-score cluster."""
    best = df[score_col].max()
    if best <= -999:
        return None
    threshold = best * pct
    plateau = df[(df[score_col] >= threshold) & (df["n_trades"] >= min_trades)].copy()
    if plateau.empty:
        plateau = df.nlargest(1, score_col)
    plateau = plateau.sort_values("ema_fast")
    mid = len(plateau) // 2
    return plateau.iloc[mid]


def main():
    parser = argparse.ArgumentParser(description="Pick middle EMA from plateau")
    parser.add_argument("csv", type=str, help="Path to optimizer CSV")
    parser.add_argument("--ini", action="store_true", help="Output as [STRATEGY:*] INI section")
    parser.add_argument("--score-col", default="wf_score", help="Score column name")
    parser.add_argument("--pct", type=float, default=0.90, help="Plateau threshold fraction of best")
    parser.add_argument("--min-trades", type=int, default=3, help="Minimum trades to qualify")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    rec = pick_middle_of_plateau(df, args.score_col, args.pct, args.min_trades)
    if rec is None:
        print("No valid params found")
        sys.exit(1)

    symbol = Path(args.csv).stem.replace("optimize_", "").replace("_", ".")
    if args.ini:
        print(f"[STRATEGY:{symbol}]")
        print("ma_type = kama")
        print(f"ema_fast_period = {int(rec['ema_fast'])}")
        print(f"ema_slow_period = {int(rec['ema_slow'])}")
        print(f"atr_sl_multiplier = {rec['sl']:.1f}")
        print(f"risk_reward_ratio = {rec['rr']:.1f}")
        print(f"adx_trend_threshold = {int(rec['adx'])}")
        print("risk_percent = 1.0")
        print(f"scoring_min_entry = {rec['score']:.2f}")
    else:
        print(f"{symbol}: EMA{int(rec['ema_fast'])}/{int(rec['ema_slow'])} "
              f"SL={rec['sl']:.1f} RR={rec['rr']:.1f} ADX={int(rec['adx'])} "
              f"score={rec['score']:.2f} WF={rec[args.score_col]:.1f}")


if __name__ == "__main__":
    main()
