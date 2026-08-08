"""Multiple-testing corrections for optimizer output (Bailey / Lopez de Prado).

A walk-forward backtest reports the performance of the *winner* of a parameter
search, but says nothing about how many configurations were tried to find it.
Under enough skill-less trials the expected maximum Sharpe grows without bound,
so an uncorrected walk-forward score is not evidence of edge.

This module computes the three numbers that make that search cost visible, and
which the optimizer writes alongside every candidate:

  effective_trials  Nominal trial counts overstate the search: neighbouring MA
                    pairs are reparameterizations of one idea and produce
                    near-identical return series. The eigenvalue participation
                    ratio of the trial-score correlation structure gives the
                    number of mutually independent bets. Published work finds
                    nominal and effective breadth can differ by one to two
                    orders of magnitude.

  expected_max_sr   E[max Sharpe] under the null that every trial has zero true
                    edge. This is the bar a result must clear to be interesting
                    at all -- not zero.

  dsr               Deflated Sharpe Ratio: the probability the true Sharpe is
                    above zero once trial count, dispersion across trials,
                    sample length and non-normality are charged against it.
                    Read as a probability; ~0.95 is the usual bar.

  pbo               Probability of Backtest Overfitting via combinatorially
                    symmetric cross-validation: how often the in-sample winner
                    lands in the bottom half out-of-sample. Above ~0.5 means
                    the selection process is picking noise.

These are diagnostics, not gates -- nothing here changes which parameters the
optimizer selects. They exist so an unusable result is visibly unusable.
"""

from __future__ import annotations

import numpy as np

# Euler-Mascheroni constant, used in the expected-maximum order statistic.
_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency)."""
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Accurate to ~1.15e-9 over the open interval, which is far beyond what these
    diagnostics need. Avoids pulling scipy into the optimizer workers.
    """
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    )


def effective_trials(combos) -> float:
    """Estimate the number of mutually *independent* configurations in a grid.

    Nominal trial counts overstate how much was really searched: neighbouring
    parameter sets are reparameterizations of one idea and produce near
    identical return series. Published work finds 136 MA lookbacks collapse to
    roughly 3 independent bets, so nominal and effective breadth can differ by
    one to two orders of magnitude.

    The redundancy lives in the grid geometry, so that is what we measure.
    *combos* is an iterable of (ema_fast, ema_slow, sl, rr, adx, score) tuples.
    Two MA pairs with the same fast/slow ratio explore essentially the same
    signal, so MA breadth is counted as the number of DISTINCT RATIOS rather
    than the number of pairs. The remaining axes (SL, RR, ADX, entry score) are
    genuinely independent dimensions and are counted by their distinct values.

    Note this deliberately does NOT infer independence from the distribution of
    result scores: score concentration is a different quantity, and treating it
    as breadth would misreport the search cost.

    Returns a float in [1, N].
    """
    combos = list(combos)
    n = len(combos)
    if n <= 1:
        return float(max(n, 1))
    try:
        ratios, sls, rrs, adxs, scs = set(), set(), set(), set(), set()
        for c in combos:
            ef, es, sl, rr, adx, sc = c[0], c[1], c[2], c[3], c[4], c[5]
            if es:
                ratios.add(round(float(ef) / float(es), 2))
            sls.add(round(float(sl), 3))
            rrs.add(round(float(rr), 3))
            adxs.add(int(adx))
            scs.add(round(float(sc), 3))
        eff = len(ratios) * len(sls) * len(rrs) * len(adxs) * len(scs)
        return float(min(max(eff, 1), n))
    except (TypeError, IndexError, ValueError):
        # Unrecognized combo shape: fall back to the nominal count, which is
        # the conservative choice (it raises the significance bar).
        return float(n)


def expected_max_sharpe(n_trials: float, sr_std: float) -> float:
    """E[max Sharpe] across *n_trials* skill-less trials with dispersion *sr_std*.

    Uses the standard Gumbel-based order-statistic approximation. Grows without
    bound in n_trials, which is the whole point: more search demands a higher
    bar, and a backtest that does not control for the extent of its search
    cannot be trusted however strong it looks.
    """
    n = max(float(n_trials), 2.0)
    if sr_std <= 0 or not np.isfinite(sr_std):
        return 0.0
    a = _norm_ppf(1.0 - 1.0 / n)
    b = _norm_ppf(1.0 - 1.0 / (n * np.e))
    return float(sr_std * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b))


def deflated_sharpe(
    observed_sr: float,
    sr_std: float,
    n_trials: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio -- P(true SR > 0) after multiple-testing correction.

    All Sharpe inputs must share the same periodicity (this codebase passes
    per-trade values). Returns a probability in [0, 1]; ~0.95 is the usual
    acceptance bar.
    """
    if n_obs < 2 or not np.isfinite(observed_sr):
        return 0.0
    sr0 = expected_max_sharpe(n_trials, sr_std)
    denom = 1.0 - skew * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if denom <= 0:
        return 0.0
    z = (observed_sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)
    if not np.isfinite(z):
        return 0.0
    return float(_norm_cdf(z))


def probability_of_backtest_overfitting(is_scores, oos_scores) -> float:
    """PBO -- how often the in-sample winner underperforms the OOS median.

    Takes paired in-sample and out-of-sample scores across the trial set. A
    value above ~0.5 means the selection procedure is systematically picking
    configurations fitted to noise; near 0 means the ranking carries over.

    This is a lightweight rank-based estimator over the supplied pairs rather
    than the full CSCV combinatorial split, which would require re-running the
    grid across every partition.
    """
    is_a = np.asarray(is_scores, dtype=float)
    oos_a = np.asarray(oos_scores, dtype=float)
    mask = np.isfinite(is_a) & np.isfinite(oos_a)
    is_a, oos_a = is_a[mask], oos_a[mask]
    n = len(is_a)
    if n < 4:
        return float("nan")

    # CSCV in spirit: repeatedly split the trial set, pick the winner on one
    # half's IS scores, and record where that winner's OOS score ranks among
    # all trials. PBO is the frequency with which the winner lands in the
    # bottom half OOS. Using many random splits keeps the statistic continuous
    # instead of the 0/1 indicator a single split would give.
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    n_splits = 64
    below = 0
    counted = 0
    for _ in range(n_splits):
        rng.shuffle(idx)
        half = idx[: max(2, n // 2)]
        winner = half[int(np.argmax(is_a[half]))]
        # Relative rank of the winner's OOS score across the full trial set.
        rank = float(np.mean(oos_a < oos_a[winner]))
        below += 1 if rank < 0.5 else 0
        counted += 1
    if counted == 0:
        return float("nan")
    return float(below / counted)


def summarize(candidate_scores, oos_scores=None, n_obs: int = 0, combos=None) -> dict:
    """Bundle the diagnostics for one symbol's optimizer run.

    *candidate_scores* are the per-trial walk-forward scores, *oos_scores* the
    matching out-of-sample scores when available, *n_obs* the number of
    observations (trades) behind the selected configuration, and *combos* the
    parameter tuples actually searched (used to discount correlated trials).

    When *combos* is omitted the nominal trial count is used, which is the
    conservative choice: it charges the full search against the result.
    """
    s = np.asarray([x for x in candidate_scores if np.isfinite(x)], dtype=float)
    if len(s) == 0:
        return {
            "n_trials": 0,
            "effective_trials": 0.0,
            "sr_std": 0.0,
            "expected_max_sr": 0.0,
            "dsr": 0.0,
            "pbo": float("nan"),
        }
    sr_std = float(np.std(s, ddof=1)) if len(s) > 1 else 0.0
    eff = effective_trials(combos) if combos is not None else float(len(s))
    observed = float(np.max(s))
    return {
        "n_trials": int(len(s)),
        "effective_trials": round(eff, 2),
        "sr_std": round(sr_std, 4),
        "expected_max_sr": round(expected_max_sharpe(eff, sr_std), 4),
        "dsr": round(deflated_sharpe(observed, sr_std, eff, max(n_obs, 2)), 4),
        "pbo": (
            round(probability_of_backtest_overfitting(s, oos_scores), 4)
            if oos_scores is not None
            else float("nan")
        ),
    }
