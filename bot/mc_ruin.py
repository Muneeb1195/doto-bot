#!/usr/bin/env python3
"""Monte Carlo ruin / drawdown estimation for Doto MT5 trade sequences.

Usage:
    python mc_ruin.py --pnls 100 50 -20 30 ...               # inline PnLs
    python mc_ruin.py --file trades.csv                       # CSV with pnl column
    python mc_ruin.py --file ../logs/backtest_*.csv --sims 10000
"""

import argparse

import numpy as np


def compute_drawdowns(equity):
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / np.maximum(running_max, 1e-10)
    return dd


def simulate(pnls, n_trades, n_simulations=10000, seed=42):
    rng = np.random.default_rng(seed)
    dd_10 = np.zeros(n_simulations, dtype=bool)
    dd_15 = np.zeros(n_simulations, dtype=bool)
    dd_20 = np.zeros(n_simulations, dtype=bool)
    dd_30 = np.zeros(n_simulations, dtype=bool)
    final_equity = np.zeros(n_simulations)
    max_dd_depth = np.zeros(n_simulations)
    for i in range(n_simulations):
        sampled = rng.choice(pnls, size=n_trades, replace=True)
        equity = 10000 + np.cumsum(sampled)
        dd = compute_drawdowns(equity)
        max_dd_depth[i] = dd.max() * 100
        dd_10[i] = (dd >= 0.10).any()
        dd_15[i] = (dd >= 0.15).any()
        dd_20[i] = (dd >= 0.20).any()
        dd_30[i] = (dd >= 0.30).any()
        final_equity[i] = equity[-1]
    return {
        "dd_10_pct": float(dd_10.mean() * 100),
        "dd_15_pct": float(dd_15.mean() * 100),
        "dd_20_pct": float(dd_20.mean() * 100),
        "dd_30_pct": float(dd_30.mean() * 100),
        "max_dd_mean": float(max_dd_depth.mean()),
        "max_dd_median": float(np.median(max_dd_depth)),
        "max_dd_95pctl": float(np.percentile(max_dd_depth, 95)),
        "final_equity_5pctl": float(np.percentile(final_equity, 5)),
        "final_equity_median": float(np.median(final_equity)),
        "final_equity_95pctl": float(np.percentile(final_equity, 95)),
        "pct_profitable": float((final_equity > 10000).mean() * 100),
    }


def kelly_fraction(pnls):
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    win_rate = len(wins) / len(pnls)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    b = avg_win / avg_loss if avg_loss > 0 else 1
    kelly = (win_rate * b - (1 - win_rate)) / b if b > 0 else 0
    return max(0.0, kelly)


def run(pnls, sims=10000):
    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    print(f"\n{'=' * 70}")
    print("  MONTE CARLO RUIN ESTIMATION")
    print(f"{'=' * 70}")
    print(f"  Trades:              {n}")
    print(f"  Simulations:         {sims:,}")
    print(f"  Mean trade:          Rs.{pnls.mean():+.2f}")
    print(f"  Median trade:        Rs.{np.median(pnls):+.2f}")
    print(f"  Std trade:           Rs.{pnls.std():.2f}")
    print(f"  Win rate:            {(pnls > 0).mean() * 100:.1f}%")
    print(f"  Profit factor:       {pnls[pnls > 0].sum() / max(abs(pnls[pnls < 0].sum()), 1e-10):.2f}")
    print(f"  Kelly fraction:      {kelly_fraction(pnls):.3f}")
    print(f"  Kelly 1/4 (default): {kelly_fraction(pnls) * 0.25:.3f}")
    print(f"{'─' * 70}")

    for n_trades in [100, 500, 1000, n]:
        label = f"Next {n_trades} trades" if n_trades < n else f"Next {n_trades} trades (all)"
        res = simulate(pnls, n_trades, n_simulations=sims)
        print(f"\n  {label}:")
        print(f"    P(drawdown >= 10%):   {res['dd_10_pct']:6.1f}%")
        print(f"    P(drawdown >= 15%):   {res['dd_15_pct']:6.1f}%")
        print(f"    P(drawdown >= 20%):   {res['dd_20_pct']:6.1f}%")
        print(f"    P(drawdown >= 30%):   {res['dd_30_pct']:6.1f}%")
        print(f"    Mean max DD:          {res['max_dd_mean']:6.1f}%")
        print(f"    95th %ile max DD:     {res['max_dd_95pctl']:6.1f}%")
        print(f"    Final equity (5th):   Rs.{res['final_equity_5pctl']:,.0f}")
        print(f"    Final equity (med):   Rs.{res['final_equity_median']:,.0f}")
        print(f"    Final equity (95th):  Rs.{res['final_equity_95pctl']:,.0f}")
        print(f"    % profitable paths:   {res['pct_profitable']:6.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo ruin estimation")
    parser.add_argument("--pnls", type=float, nargs="+", help="Trade PnL values")
    parser.add_argument("--file", type=str, help="CSV file with trade PnLs (must have 'pnl' column)")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations")
    args = parser.parse_args()
    pnls = None
    if args.pnls:
        pnls = args.pnls
    elif args.file:
        import pandas as pd

        df = pd.read_csv(args.file)
        if "pnl" not in df.columns:
            print("CSV must have a 'pnl' column")
            return
        pnls = df["pnl"].dropna().values
        print(f"Loaded {len(pnls)} trades from {args.file}")
    else:
        print("Provide --pnls or --file")
        return
    if len(pnls) < 10:
        print(f"Too few trades ({len(pnls)}), need at least 10")
        return
    run(pnls, args.sims)


if __name__ == "__main__":
    main()
