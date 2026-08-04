import logging

import numpy as np

logger = logging.getLogger(__name__)


def percentileofscore(arr, score):
    if len(arr) == 0:
        return 50.0
    return float((arr <= score).mean() * 100)


def monte_carlo_reshuffle(trades_pnls, n_simulations=5000):
    len(trades_pnls)
    final_pnls = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)
    sharpe_arr = np.zeros(n_simulations)
    pf_arr = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = np.random.permutation(trades_pnls)
        equity = np.cumsum(shuffled)
        final_pnls[i] = equity[-1]
        running_max = np.maximum.accumulate(equity)
        dd = running_max - equity
        max_dds[i] = dd.max() if len(dd) > 0 else 0
        returns = np.diff(equity) / (np.abs(equity[:-1]) + 1e-10)
        sharpe_arr[i] = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if np.std(returns) > 0 else 0
        wins = shuffled[shuffled > 0].sum()
        losses = abs(shuffled[shuffled < 0].sum())
        pf_arr[i] = wins / losses if losses > 0 else 999

    return final_pnls, max_dds, sharpe_arr, pf_arr


def monte_carlo_bootstrap(trades_pnls, n_simulations=5000):
    n = len(trades_pnls)
    final_pnls = np.zeros(n_simulations)
    for i in range(n_simulations):
        sampled = np.random.choice(trades_pnls, size=n, replace=True)
        final_pnls[i] = np.sum(sampled)
    return final_pnls


def compute_mc_report(trades_pnls, original_metrics, n_simulations=5000):
    final_pnls, max_dds, sharpe_arr, pf_arr = monte_carlo_reshuffle(trades_pnls, n_simulations)
    # Bootstrap too for extra validation
    boot_pnls = monte_carlo_bootstrap(trades_pnls, n_simulations // 2)

    report = {
        "n_simulations": n_simulations,
        "n_trades": len(trades_pnls),
        "original_return": original_metrics["return_pct"],
        "original_sharpe": original_metrics["sharpe"],
        "original_max_dd": original_metrics["max_dd_pct"],
        "original_pf": original_metrics["profit_factor"],
        "original_win_rate": original_metrics["win_rate"],
        "mc_median_return": np.median(final_pnls),
        "mc_median_sharpe": np.median(sharpe_arr),
        "mc_median_max_dd": np.median(max_dds),
        "mc_median_pf": np.median(pf_arr),
        "return_5th_pctl": float(np.percentile(final_pnls, 5)),
        "return_95th_pctl": float(np.percentile(final_pnls, 95)),
        "dd_95th_pctl": float(np.percentile(max_dds, 95)),
        "sharpe_5th_pctl": float(np.percentile(sharpe_arr, 5)),
        "sharpe_95th_pctl": float(np.percentile(sharpe_arr, 95)),
        "sharpe_percentile_rank": percentileofscore(sharpe_arr, original_metrics["sharpe"]),
        "pct_profitable_paths": float((final_pnls > 0).mean() * 100),
        "pct_profitable_bootstrap": float((boot_pnls > 0).mean() * 100),
        "dd_ratio_95pctl": float(np.percentile(max_dds, 95) / (abs(original_metrics["max_dd_pct"]) + 1e-10)),
    }
    return report


def print_mc_report(report):
    print(f"\n{'=' * 70}")
    print(f"MONTE CARLO ROBUSTNESS CHECK ({report['n_simulations']:,} simulations)")
    print(f"{'=' * 70}")
    print(f"  Trades:              {report['n_trades']}")
    print(f"  Original Return:     ${report['original_return']:+,.2f}")
    print(f"  Original Sharpe:     {report['original_sharpe']:.2f}")
    print(f"  Original Max DD:     ${report['original_max_dd']:+,.2f}")
    print(f"  Original Win Rate:   {report['original_win_rate'] * 100:.1f}%")
    print(f"  Original Profit F:   {report['original_pf']:.2f}")
    print(f"{'─' * 70}")
    print(f"  MC Median Return:    ${report['mc_median_return']:+,.2f}")
    print(f"  MC Median Sharpe:    {report['mc_median_sharpe']:.2f}")
    print(f"  MC Median Max DD:    ${report['mc_median_max_dd']:+,.2f}")
    print(f"  MC Median PF:        {report['mc_median_pf']:.2f}")
    print(f"{'─' * 70}")
    print(f"  Sharpe 5th-95th:      {report['sharpe_5th_pctl']:.2f} to {report['sharpe_95th_pctl']:.2f}")
    print(f"  Return 5th-95th:      ${report['return_5th_pctl']:+,.2f} to ${report['return_95th_pctl']:+,.2f}")
    print(f"  95th %ile DD:         ${report['dd_95th_pctl']:+,.2f}")
    print(f"  DD ratio (95th/Orig): {report['dd_ratio_95pctl']:.2f}x")
    print(f"{'─' * 70}")
    print(f"  Sharpe percentile:    {report['sharpe_percentile_rank']:.1f}%")
    print(f"  % Profitable paths:   {report['pct_profitable_paths']:.1f}%")
    print(f"  % Profitable boot:    {report['pct_profitable_bootstrap']:.1f}%")
    print(f"{'─' * 70}")

    robust = True
    if report["sharpe_percentile_rank"] < 90:
        print(
            f"  ⚠ FAIL: Sharpe rank < 90% ({report['sharpe_percentile_rank']:.1f}%) "
            f"— signal may not beat random ordering"
        )
        robust = False
    if report["pct_profitable_paths"] < 80:
        print(f"  ⚠ FAIL: Only {report['pct_profitable_paths']:.1f}% of paths profitable (need >80%)")
        robust = False
    if report["dd_ratio_95pctl"] > 2.5:
        print(f"  ⚠ FAIL: DD ratio {report['dd_ratio_95pctl']:.2f}x > 2.5x — backtest may have gotten lucky on timing")
        robust = False
    if report["n_trades"] < 20:
        print(f"  ⚠ FAIL: Only {report['n_trades']} trades — too few for statistical confidence")
        robust = False

    if robust:
        print("  ✅ PASS: Strategy is robust — deploy with confidence")
    else:
        print("  ❌ FAIL: Strategy needs more work — consider filtering or adjusting parameters")
    print()


if __name__ == "__main__":
    demo_pnls = np.random.randn(50) * 100 + 20
    demo_metrics = {
        "return_pct": float(np.sum(demo_pnls)),
        "sharpe": 1.2,
        "max_dd_pct": -500.0,
        "profit_factor": 1.8,
        "win_rate": 0.6,
    }
    report = compute_mc_report(demo_pnls, demo_metrics, n_simulations=1000)
    print_mc_report(report)
