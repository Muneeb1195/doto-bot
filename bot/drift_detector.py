import logging
import time

import numpy as np

_drift_warmstart_queue: dict[str, list] = {}
_drift_last_check: dict[str, float] = {}
_DRIFT_COOLDOWN = 3600

PSI_THRESHOLD = 0.2
CONFIDENCE_DROP_THRESHOLD = 0.15
MIN_SAMPLES_PSI = 200


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


def check_feature_psi(symbol, feature_data, feature_stats):
    if symbol not in _drift_last_check:
        _drift_last_check[symbol] = 0
    if time.time() - _drift_last_check.get(symbol, 0) < _DRIFT_COOLDOWN:
        return False
    drifted_features = 0
    max_psi = 0.0
    for col, stats in feature_stats.items():
        if col not in feature_data.columns:
            continue
        mu = stats.get("mean")
        sigma = stats.get("std")
        if mu is None or sigma is None or sigma <= 0:
            continue
        live_vals = feature_data[col].dropna().values[-MIN_SAMPLES_PSI:]
        if len(live_vals) < 50:
            continue
        train_dist = np.random.normal(mu, sigma, size=1000)
        psi = compute_psi(train_dist, live_vals)
        if psi > PSI_THRESHOLD:
            drifted_features += 1
            max_psi = max(max_psi, psi)
    if drifted_features >= 3:
        logging.warning(
            f"[{symbol}] Feature drift: {drifted_features} features exceed "
            f"PSI={PSI_THRESHOLD:.1f} (max_psi={max_psi:.3f})"
        )
        _drift_last_check[symbol] = time.time()
        return True
    return False


def check_confidence_drift(symbol, baseline_conf, recent_mean_conf):
    if baseline_conf <= 0:
        return False
    drop_ratio = 1.0 - (recent_mean_conf / baseline_conf)
    if drop_ratio > CONFIDENCE_DROP_THRESHOLD:
        logging.warning(
            f"[{symbol}] Confidence drift: baseline={baseline_conf:.3f} "
            f"recent={recent_mean_conf:.3f} drop={drop_ratio * 100:.0f}%"
        )
        return True
    return False


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
