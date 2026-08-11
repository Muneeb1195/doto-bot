import logging
import time

import numpy as np

_drift_warmstart_queue: dict[str, float] = {}
_drift_last_check: dict[str, float] = {}
_DRIFT_COOLDOWN = 3600


def compute_psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0
    all_vals = np.concatenate([expected, actual])
    if np.std(all_vals) < 1e-10:
        return 0.0
    bin_edges = np.percentile(expected, np.linspace(0, 100, bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    psi = 0.0
    for i in range(bins):
        pct_exp = np.sum((expected >= bin_edges[i]) & (expected < bin_edges[i + 1])) / len(expected)
        pct_act = np.sum((actual >= bin_edges[i]) & (actual < bin_edges[i + 1])) / len(actual)
        pct_act = max(pct_act, 1e-10)
        pct_exp = max(pct_exp, 1e-10)
        psi += (pct_act - pct_exp) * np.log(pct_act / pct_exp)
    return psi


def schedule_warmstart(symbol):
    global _drift_warmstart_queue
    _drift_warmstart_queue[symbol] = time.time()
    logging.info(f"[{symbol}] Warm-start queued (will run next idle cycle)")


def has_pending_warmstart(symbol):
    return symbol in _drift_warmstart_queue


def consume_warmstart_queue():
    global _drift_warmstart_queue
    pending = list(_drift_warmstart_queue.keys())
    _drift_warmstart_queue = {}
    return pending
