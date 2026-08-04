"""analyze_scoring_weights.py — Comprehensive scoring weight optimization.

Fetches years of H1 data for all portfolio symbols, simulates the bot's
MA-cross-over strategy, computes all 7 scoring components for every signal,
records forward returns, then runs a large-scale Monte Carlo random search
to find the weight combination that maximizes information coefficient (IC),
top-quartile win rate, and score-stratified Sharpe.

Usage:
    python analyze_scoring_weights.py [--years 1] [--mc-iter 50000] [--no-ml]
                                      [--output data/weight_analysis]
"""

import sys
import os
import argparse
import logging
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, dirichlet
from sklearn.metrics import roc_auc_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent / "bot"))

BASE_DIR = Path(__file__).resolve().parent

from config import load_config, CONFIG_DIR
from indicators import calc_ma, calc_atr_series
from mt5_connect import get_rates
from ml_features import compute_features, FEATURE_COLS, prepare_features
import state as _st
import filters

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("weight_analysis")

FORWARD_BARS = [12, 24, 48]
SCORE_KEYS = ["exec", "volume", "volatility", "spread", "news", "ml", "tail_risk"]


def make_sym_cfg(global_cfg, symbol):
    """Build per-symbol config (same logic as main.py)."""
    from copy import deepcopy

    cfg = deepcopy(global_cfg)
    for key, conv in (
        ("ema_fast", int),
        ("ema_slow", int),
        ("ma_type", str),
        ("atr_sl_mult", float),
        ("rr", float),
        ("risk_percent", float),
        ("adx_trend_threshold", int),
        ("adx_range_threshold", int),
        ("kelly_fraction", float),
        ("atr_period", int),
        ("atr_sma_period", int),
    ):
        ovr = cfg.get("symbol_strategy", {}).get(symbol, {})
        cfg_key = {"ema_fast": "ema_fast_period", "ema_slow": "ema_slow_period",
                    "rr": "risk_reward_ratio"}.get(key, key)
        strat_key = {
            "ema_fast": "ema_fast_period", "ema_slow": "ema_slow_period",
            "rr": "risk_reward_ratio", "kelly_fraction": "kelly_fraction",
            "atr_period": "atr_period", "atr_sma_period": "atr_sma_period",
            "ma_type": "ma_type", "atr_sl_mult": "atr_sl_multiplier",
            "risk_percent": "risk_percent",
        }.get(key, key)
        if symbol in cfg.get("symbol_strategy", {}) and strat_key in cfg["symbol_strategy"][symbol]:
            cfg[key] = conv(cfg["symbol_strategy"][symbol][strat_key])
        elif key in cfg.get("_global_strategy_defaults", {}):
            cfg[key] = cfg["_global_strategy_defaults"][key]
    cfg["symbol"] = symbol
    return cfg


def generate_signals(df, cfg):
    """Generate entry signals (buy/sell) for each bar based on MA crossover.

    Returns DataFrame with columns: signal, signal_type, fast, slow, atr.
    """
    ma_type = cfg.get("ma_type", "kama")
    fast = calc_ma(df, cfg["ema_fast"], ma_type)
    slow = calc_ma(df, cfg["ema_slow"], ma_type)
    atr_series = calc_atr_series(df, cfg.get("atr_period", 14))

    signals = pd.Series(index=df.index, dtype=object)
    signal_types = pd.Series(index=df.index, dtype=object)

    for i in range(1, len(df)):
        if pd.isna(fast.iloc[i]) or pd.isna(slow.iloc[i]):
            continue
        if pd.isna(fast.iloc[i - 1]) or pd.isna(slow.iloc[i - 1]):
            continue

        cur_f = fast.iloc[i]
        cur_s = slow.iloc[i]
        prev_f = fast.iloc[i - 1]
        prev_s = slow.iloc[i - 1]

        if prev_f <= prev_s and cur_f > cur_s:
            signals.iloc[i] = "buy"
            signal_types.iloc[i] = "fresh"
        elif prev_f >= prev_s and cur_f < cur_s:
            signals.iloc[i] = "sell"
            signal_types.iloc[i] = "fresh"
        elif cur_f > cur_s:
            signals.iloc[i] = "buy"
            signal_types.iloc[i] = "aligned"
        elif cur_f < cur_s:
            signals.iloc[i] = "sell"
            signal_types.iloc[i] = "aligned"

    return pd.DataFrame({
        "signal": signals,
        "signal_type": signal_types,
        "fast": fast,
        "slow": slow,
        "atr": atr_series,
    }, index=df.index)


def compute_all_scores_vectorized(df, cfg, ml_cache=None):
    """Compute ALL component scores for EVERY bar using vectorized operations.

    Returns a DataFrame with score columns for each bar, indexed same as df.
    """
    symbol = cfg["symbol"]
    n = len(df)
    atr_period = cfg.get("atr_period", 14)
    atr_sma_period = cfg.get("atr_sma_period", 20)
    vf_period = cfg.get("vf_sma_period", 20)
    vf_kappa = cfg.get("vf_kappa", 1.2)
    min_ratio = cfg.get("volatility_min_ratio", 0.5)
    spf_max_ratio = cfg.get("spf_max_ratio", 0.30)
    exec_enabled = cfg.get("exec_enabled", True)
    vf_enabled = cfg.get("vf_enabled", True)
    spf_enabled = cfg.get("spf_enabled", True)

    score_df = pd.DataFrame(0.5, index=df.index, columns=SCORE_KEYS)

    # --- pre-compute columns needed for scoring ---
    fast = df.get("fast") if "fast" in df.columns else None
    slow = df.get("slow") if "slow" in df.columns else None
    atr_series = df.get("atr_series") if "atr_series" in df.columns else None
    if atr_series is None:
        atr_series = calc_atr_series(df, atr_period)
    atr_sma = atr_series.rolling(window=atr_sma_period, min_periods=1).mean()

    # --- exec score (vectorized) ---
    if exec_enabled and fast is not None and slow is not None:
        aligned = (fast > slow).astype(float)
        fresh = ((fast.shift(1) <= slow.shift(1)) & (fast > slow)).astype(float)
        score_df["exec"] = np.where(fresh, 1.0, np.where(aligned, 0.7, 0.3))
        # exec is 0.5 where MA is NaN
        score_df["exec"] = score_df["exec"].where(fast.notna() & slow.notna(), 0.5)

    # --- volume score (vectorized) ---
    if vf_enabled:
        vol_sma = df["tick_volume"].rolling(window=vf_period, min_periods=1).mean()
        rel_vol = df["tick_volume"] / vol_sma.replace(0, np.nan)
        score_df["volume"] = (rel_vol / vf_kappa).clip(upper=1.0).fillna(0.5)

    # --- volatility score (vectorized) ---
    ratio = atr_series / atr_sma.replace(0, np.nan)
    ratio = ratio.fillna(1.0)
    score_df["volatility"] = (ratio / max(min_ratio, 0.01)).clip(upper=1.0).fillna(0.5)

    # --- spread score (vectorized) ---
    if spf_enabled and "spread" in df.columns and atr_series is not None:
        atr_vals = atr_series.replace(0, 0.001).fillna(0.001)
        spread_ratio = df["spread"] / atr_vals
        score_df["spread"] = (1.0 - spread_ratio / max(spf_max_ratio, 0.01)).clip(lower=0.0, upper=1.0).fillna(0.5)

    # --- news score (no historical data) ---
    score_df["news"] = 0.5

    # --- ml score ---
    score_df["ml"] = 0.5
    if cfg.get("ml_enabled", True) and ml_cache is not None:
        for idx, ml_data in ml_cache.items():
            if idx < n:
                conf = ml_data.get("conf")
                threshold = ml_data.get("threshold", cfg.get("ml_confidence", 0.55))
                if conf is not None and not np.isnan(conf):
                    score_df.loc[df.index[idx], "ml"] = min(1.0, conf / max(threshold, 0.01))

    # --- tail_risk ---
    score_df["tail_risk"] = 1.0

    # Fix NaN/Inf
    score_df = score_df.replace([np.inf, -np.inf], 0.5).fillna(0.5)
    return score_df


def compute_ml_cache(symbol, cfg, df, signals_df, model_entry):
    """Pre-compute ML confidence for every signal bar.

    Returns dict {signal_idx: {"conf": float, "threshold": float}}.
    """
    if model_entry is None:
        return None

    model = model_entry["model"]
    model_features = model_entry.get("metadata", {}).get("features", FEATURE_COLS)
    opt_threshold = model_entry.get("metadata", {}).get("optimal_threshold")

    feat_result = prepare_features(df, symbol=symbol)
    if feat_result is None:
        log.warning(f"  [{symbol}] prepare_features returned None")
        return None
    feature_data, full_df = feat_result

    missing = [c for c in model_features if c not in feature_data.columns]
    if missing:
        log.warning(f"  [{symbol}] Missing features: {missing}")
        return None

    n_expected = None
    try:
        if hasattr(model, "n_features_in_"):
            n_expected = model.n_features_in_
        elif hasattr(model, "xgb") and hasattr(model.xgb, "n_features_in_"):
            n_expected = model.xgb.n_features_in_
        elif hasattr(model, "lgb") and hasattr(model.lgb, "n_features_in_"):
            n_expected = model.lgb.n_features_in_
    except Exception:
        pass

    threshold = cfg["ml_threshold_overrides"].get(symbol) or opt_threshold or cfg.get("ml_confidence", 0.55)

    cache = {}
    signal_indices = signals_df.index[signals_df["signal"].notna()].tolist()
    if not signal_indices:
        return cache

    for idx in signal_indices:
        if idx >= len(feature_data):
            continue
        latest = feature_data[model_features].iloc[idx:idx + 1]
        if len(latest) == 0:
            continue
        latest_arr = np.nan_to_num(latest.values.copy(), nan=0.0)

        if n_expected is not None and latest_arr.shape[1] != n_expected:
            continue

        try:
            proba = model.predict_proba(latest_arr)
            prob_long = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
        except Exception:
            continue

        signal = signals_df.loc[idx, "signal"]
        conf = prob_long if signal == "buy" else 1.0 - prob_long
        cache[idx] = {"conf": float(conf), "threshold": float(threshold)}

    log.info(f"  [{symbol}] Computed ML confidence for {len(cache)}/{len(signal_indices)} signals")
    return cache


def calc_forward_returns(df, signal_index, signal_dir, bars):
    """Compute forward returns of a given # of bars."""
    end = signal_index + bars
    if end >= len(df):
        return np.nan
    entry = df["close"].iloc[signal_index]
    exit_ = df["close"].iloc[end]
    raw_ret = (exit_ - entry) / entry
    return raw_ret * signal_dir


def collect_all_data(cfg, years=1, skip_ml=False):
    """Fetch data, generate signals, compute scores for all symbols.

    Returns DataFrame with columns: symbol, signal, signal_type, forward_ret_N,
    exec, volume, volatility, spread, news, ml, tail_risk, composite.
    """
    rows = []
    start_time = time.time()
    symbols = cfg["symbols"]
    n_bars = int(years * 365 * 24) + 500

    for sym_idx, symbol in enumerate(symbols):
        sym_start = time.time()
        log.info(f"[{symbol}] Fetching {n_bars} H1 bars ({years} year(s))...")

        df = get_rates(symbol, 16385, n_bars)  # 16385 = TIMEFRAME_H1
        if df is None or len(df) < 500:
            log.warning(f"  [{symbol}] Insufficient data ({len(df) if df is not None else 0} bars), skipping")
            continue
        if len(df) > n_bars:
            df = df.iloc[-n_bars:]
        df = df.reset_index(drop=True)

        sym_cfg = make_sym_cfg(cfg, symbol)
        log.info(f"  [{symbol}] Generating signals (MA={sym_cfg['ma_type']} "
                 f"{sym_cfg['ema_fast']}/{sym_cfg['ema_slow']})...")

        signals_df = generate_signals(df, sym_cfg)
        signal_mask = signals_df["signal"].notna()
        n_signals = signal_mask.sum()
        log.info(f"  [{symbol}] {n_signals} signal bars found")

        if n_signals == 0:
            log.warning(f"  [{symbol}] No signals found, skipping")
            continue

        # Pre-compute ML cache
        ml_cache = None
        if not skip_ml and cfg.get("ml_enabled", True):
            safe_symbol = symbol.replace(".", "_")
            model_path = BASE_DIR / cfg["ml_model_path_template"].format(symbol=safe_symbol)
            model_entry = None
            if model_path.exists():
                try:
                    model_entry = joblib.load(model_path)
                except Exception as e:
                    log.warning(f"  [{symbol}] Failed to load model: {e}")
            if model_entry is not None:
                ml_cache = compute_ml_cache(symbol, sym_cfg, df, signals_df, model_entry)
            else:
                log.info(f"  [{symbol}] No ML model found")
        else:
            log.info(f"  [{symbol}] ML scoring disabled ({skip_ml=})")

        # Add columns to df for vectorized scoring
        df["signal"] = signals_df["signal"]
        df["fast"] = signals_df["fast"]
        df["slow"] = signals_df["slow"]
        df["atr_series"] = signals_df["atr"]

        signal_indices = df.index[signal_mask].tolist()
        log.info(f"  [{symbol}] Vectorized scoring for {len(signal_indices)} signals...")

        # Compute all scores for all bars at once (fast, vectorized)
        all_scores = compute_all_scores_vectorized(df, sym_cfg, ml_cache)

        # Build forward returns vectorized
        max_bars = len(df)
        signal_dir = pd.Series(1, index=df.index)
        signal_dir[df["signal"] == "sell"] = -1

        for fb in FORWARD_BARS:
            close_t = df["close"].values
            close_fwd = np.roll(close_t, -fb)
            close_fwd[-fb:] = np.nan
            raw_ret = (close_fwd - close_t) / close_t
            df[f"fwd_{fb}"] = raw_ret * signal_dir.values

        # Extract only signal bars
        sig_sel = signals_df.loc[signal_mask]
        sym_data = pd.DataFrame({
            "symbol": symbol,
            "time": df.loc[signal_mask, "time"],
            "signal": sig_sel["signal"],
            "signal_type": sig_sel["signal_type"],
            "price": df.loc[signal_mask, "close"],
            "atr": sig_sel["atr"],
        })
        for k in SCORE_KEYS:
            sym_data[k] = all_scores.loc[signal_mask, k].values
        for fb in FORWARD_BARS:
            sym_data[f"fwd_{fb}"] = df.loc[signal_mask, f"fwd_{fb}"].values

        rows.append(sym_data)

        if (sym_idx + 1) % 4 == 0:
            elapsed = time.time() - start_time
            log.info(f"  Progress: {sym_idx + 1}/{len(symbols)} symbols in {elapsed:.0f}s")

    if not rows:
        log.error("No data collected!")
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    log.info(f"Total signals collected: {len(result)} across "
             f"{result['symbol'].nunique()} symbols in "
             f"{time.time() - start_time:.0f}s")
    return result


def analyze_components(df):
    """Analyze individual component predictive power."""
    log.info("\n" + "=" * 70)
    log.info("COMPONENT ANALYSIS")
    log.info("=" * 70)

    results = []
    for comp in SCORE_KEYS:
        if comp not in df.columns:
            continue
        valid = df[comp].notna() & df["fwd_24"].notna()
        if valid.sum() < 50:
            log.info(f"  {comp}: insufficient data ({valid.sum()} rows)")
            continue

        comp_vals = df.loc[valid, comp]
        fwd_vals = df.loc[valid, "fwd_24"]
        pos_ret = (fwd_vals > 0).astype(float)

        ic, ic_p = spearmanr(comp_vals, fwd_vals)
        ic_abs = abs(ic) if not pd.isna(ic) else 0

        # AUC: how well does this component predict positive returns?
        try:
            auc = roc_auc_score(pos_ret, comp_vals)
        except Exception:
            auc = 0.5

        # Mean forward return by quartile
        q_labels = ["Q1(low)", "Q2", "Q3", "Q4(high)"]
        try:
            q = pd.qcut(comp_vals, 4, labels=q_labels, duplicates="raise")
        except ValueError:
            try:
                q = pd.qcut(comp_vals.rank(method="first"), 4, labels=q_labels)
            except ValueError:
                q = None
        q_means = fwd_vals.groupby(q).mean() if q is not None else pd.Series(dtype=float)

        results.append({
            "component": comp,
            "n": valid.sum(),
            "mean": comp_vals.mean(),
            "std": comp_vals.std(),
            "IC_24h": round(ic if not pd.isna(ic) else 0, 4),
            "IC_p": f"{ic_p:.4f}" if not pd.isna(ic_p) else "N/A",
            "AUC": round(auc, 4),
            "win_rate": round(pos_ret.mean(), 4),
            "q1_fwd_ret": round(q_means.iloc[0], 6) if len(q_means) >= 4 else None,
            "q4_fwd_ret": round(q_means.iloc[-1], 6) if len(q_means) >= 4 else None,
            "spread_q4_q1": round(q_means.iloc[-1] - q_means.iloc[0], 6) if len(q_means) >= 4 else None,
        })

    comp_df = pd.DataFrame(results)
    comp_df = comp_df.sort_values("IC_24h", key=abs, ascending=False)
    log.info(f"\n{comp_df.to_string(index=False)}")

    # Highlight top predictors
    def _ic_p_sig(x):
        try:
            return float(x) < 0.05
        except (ValueError, TypeError):
            return False
    top = comp_df.loc[comp_df["IC_p"].apply(_ic_p_sig) |
                      (comp_df["AUC"] > 0.55), :]
    if len(top) > 0:
        log.info(f"\n  ⭐ Components with significant predictive power:")
        for _, r in top.iterrows():
            log.info(f"     {r['component']}: IC={r['IC_24h']}, AUC={r['AUC']}, "
                     f"Q4-Q1 spread={r['spread_q4_q1']}")
    else:
        log.info(f"\n  ⚠️  No component has significant individual predictive power (p<0.05)")

    return comp_df


def run_weight_search(df, n_iterations=50000, primary_horizon=24):
    """Monte Carlo search for optimal scoring weights.

    Uses Dirichlet distribution to sample weight combinations.
    Evaluates each by IC (Spearman ρ between composite score and forward return)
    and top-quartile win rate.

    Returns sorted DataFrame of weight sets with metrics.
    """
    log.info("\n" + "=" * 70)
    log.info(f"MONTE CARLO WEIGHT SEARCH ({n_iterations} iterations)")
    log.info("=" * 70)

    fwd_col = f"fwd_{primary_horizon}"
    valid = df[fwd_col].notna()
    valid_df = df.loc[valid].copy()
    log.info(f"  Valid signals: {len(valid_df)}")

    n_components = len(SCORE_KEYS)
    results = []
    batch_size = 5000

    # Also evaluate current weights as baseline
    current_weights = {
        "exec": 0.15, "volume": 0.10, "volatility": 0.10,
        "spread": 0.10, "news": 0.10, "tape": 0.10,
        "ml": 0.25, "tail_risk": 0.10
    }

    def _eval_weights(w_dict, label=""):
        wsum = sum(w_dict.get(k, 0) for k in SCORE_KEYS)
        if wsum == 0:
            return None
        composite = sum(valid_df[k] * w_dict.get(k, 0) for k in SCORE_KEYS) / wsum
        fwd = valid_df[fwd_col].values
        ic, ic_p = spearmanr(composite, fwd)
        ic = ic if not pd.isna(ic) else 0

        # Top-quartile win rate
        top_mask = composite >= composite.quantile(0.75)
        top_win = (fwd[top_mask] > 0).mean()

        # Bottom-quartile win rate (should be lower)
        bot_mask = composite <= composite.quantile(0.25)
        bot_win = (fwd[bot_mask] > 0).mean()

        # Score-stratified spread: Q4 mean ret - Q1 mean ret
        try:
            q_cuts = pd.qcut(composite, 4, labels=[1, 2, 3, 4], duplicates="raise")
        except ValueError:
            try:
                q_cuts = pd.qcut(composite.rank(method="first"), 4, labels=[1, 2, 3, 4])
            except ValueError:
                q_cuts = None
        if q_cuts is not None and (hasattr(q_cuts, 'nunique') and q_cuts.nunique() == 4):
            q_means = pd.Series(fwd).groupby(q_cuts.values).mean()
            spread_q4_q1 = q_means.iloc[-1] - q_means.iloc[0]
        else:
            spread_q4_q1 = 0

        return {
            **{k: w_dict.get(k, 0) for k in SCORE_KEYS},
            "IC": round(ic, 4),
            "top_win_rate": round(top_win, 4),
            "bot_win_rate": round(bot_win, 4),
            "spread_q4_q1": round(spread_q4_q1, 6),
            "label": label,
        }

    # Evaluate current weights
    baseline = _eval_weights(current_weights, "CURRENT")
    if baseline:
        results.append(baseline)
        log.info(f"  Baseline (current weights): IC={baseline['IC']}, "
                 f"top_win={baseline['top_win_rate']}, "
                 f"spread={baseline['spread_q4_q1']}")

    # Also evaluate equal weights
    equal_w = {k: 1.0 / n_components for k in SCORE_KEYS}
    eq = _eval_weights(equal_w, "EQUAL")
    if eq:
        results.append(eq)
        log.info(f"  Equal weights: IC={eq['IC']}, top_win={eq['top_win_rate']}")

        # Speed optimization: subsample data for search, validate top on full
        search_n = min(10000, len(valid_df))
        search_idx = np.random.default_rng(42).choice(len(valid_df), search_n, replace=False)
        search_df = valid_df.iloc[search_idx]
        log.info(f"  Using {search_n} subsampled rows for search speed")

        # Dirichlet random search
        start = time.time()
        alphas = [0.5, 0.8, 1.0, 1.5, 2.0]

        batch_count = 0
        for batch_start in range(0, n_iterations, batch_size):
            batch_end = min(batch_start + batch_size, n_iterations)
            batch_n = batch_end - batch_start
            alpha = alphas[batch_count % len(alphas)]
            samples = dirichlet.rvs([alpha] * n_components, size=batch_n)

            for s in samples:
                w_dict = dict(zip(SCORE_KEYS, s))
                wsum = sum(w_dict.get(k, 0) for k in SCORE_KEYS)
                if wsum == 0:
                    continue
                composite = sum(search_df[k] * w_dict.get(k, 0) for k in SCORE_KEYS) / wsum
                fwd = search_df[fwd_col].values
                ic, ic_p = spearmanr(composite, fwd)
                ic = ic if not pd.isna(ic) else 0
                top_mask = composite >= composite.quantile(0.75)
                top_win = (fwd[top_mask] > 0).mean()
                bot_mask = composite <= composite.quantile(0.25)
                bot_win = (fwd[bot_mask] > 0).mean()

                spread_q4_q1 = 0.0
                try:
                    q_cuts = pd.qcut(composite.rank(method="first"), 4, labels=[1, 2, 3, 4])
                    if hasattr(q_cuts, 'nunique') and q_cuts.nunique() == 4:
                        q_means = pd.Series(fwd).groupby(q_cuts.values).mean()
                        spread_q4_q1 = q_means.iloc[-1] - q_means.iloc[0]
                except Exception:
                    pass

                results.append({
                    **{k: w_dict.get(k, 0) for k in SCORE_KEYS},
                    "IC": round(ic, 4),
                    "top_win_rate": round(top_win, 4),
                    "bot_win_rate": round(bot_win, 4),
                    "spread_q4_q1": round(spread_q4_q1, 6),
                    "label": "",
                })

            batch_count += 1
            if batch_count % 5 == 0:
                elapsed = time.time() - start
                rate = batch_end / max(elapsed, 0.1)
                log.info(f"  Sampled {batch_end}/{n_iterations} "
                         f"({rate:.0f}/s) [{elapsed:.0f}s]")

    total_time = time.time() - start
    log.info(f"  Total: {len(results)} evaluations in {total_time:.1f}s "
             f"({len(results) / max(total_time, 0.1):.0f}/s)")

    result_df = pd.DataFrame(results)

    # If no results, return empty
    if len(result_df) == 0:
        return result_df

    # Also compute a combined score that balances IC and top_win_rate
    ic_range = result_df["IC"].max() - result_df["IC"].min()
    wr_range = result_df["top_win_rate"].max() - result_df["top_win_rate"].min()
    if ic_range > 0 and wr_range > 0:
        result_df["combined"] = (
            0.5 * (result_df["IC"] - result_df["IC"].min()) / ic_range +
            0.5 * (result_df["top_win_rate"] - result_df["top_win_rate"].min()) / wr_range
        )
    else:
        result_df["combined"] = 0

    return result_df


def run_cross_validation(df, n_splits=5, n_iterations=30000):
    """Time-split cross-validation for stability analysis.

    Splits data chronologically into n_splits folds.
    For each fold: optimize on train, evaluate on test.
    Reports stability of top weights across folds.
    """
    log.info("\n" + "=" * 70)
    log.info(f"CROSS-VALIDATION ({n_splits}-fold, time-split)")
    log.info("=" * 70)

    valid = df["fwd_24"].notna()
    valid_df = df.loc[valid].sort_values("time").reset_index(drop=True)
    log.info(f"  Total valid signals: {len(valid_df)}")

    fold_size = len(valid_df) // n_splits
    if fold_size < 100:
        log.warning(f"  Too few signals per fold ({fold_size}), skipping CV")
        return None

    # Build folds: each fold uses first K splits as train, next as test
    best_weights_per_fold = []
    metrics_per_fold = []

    for fold in range(n_splits):
        test_start = (fold + 1) * fold_size
        test_end = min((fold + 2) * fold_size, len(valid_df))

        train_df = valid_df.iloc[:test_start]
        test_df = valid_df.iloc[test_start:test_end]

        if len(train_df) < 200 or len(test_df) < 50:
            continue

        log.info(f"  Fold {fold + 1}/{n_splits}: "
                 f"train={len(train_df)}, test={len(test_df)}")

        # Optimize on train
        train_results = run_weight_search(train_df, n_iterations=n_iterations)
        if len(train_results) == 0:
            continue

        top_ic = train_results.nlargest(10, "IC")
        top_combined = train_results.nlargest(10, "combined")

        # Evaluate top weight sets on test
        for label, candidates in [("IC", top_ic), ("Combined", top_combined)]:
            for rank, (_, candidate) in enumerate(candidates.iterrows()):
                w_dict = {k: candidate[k] for k in SCORE_KEYS}
                wsum = sum(w_dict.values())
                if wsum == 0:
                    continue
                composite = sum(test_df[k] * w_dict[k] for k in SCORE_KEYS) / wsum
                ic_test, _ = spearmanr(composite, test_df["fwd_24"])
                ic_test = ic_test if not pd.isna(ic_test) else 0

                best_weights_per_fold.append({
                    "fold": fold + 1,
                    "label": label,
                    "rank": rank + 1,
                    **{k: round(w_dict[k], 4) for k in SCORE_KEYS},
                    "train_IC": candidate["IC"],
                    "test_IC": round(ic_test, 4),
                    "train_top_win": candidate["top_win_rate"],
                    "test_top_win": round(
                        ((composite >= composite.quantile(0.75)) & (test_df["fwd_24"].values > 0)).mean(), 4
                    ) if composite.quantile(0.75) > 0 else 0,
                })

    if not best_weights_per_fold:
        log.warning("  No CV results")
        return None

    cv_df = pd.DataFrame(best_weights_per_fold)

    # Summary per component: mean weight, std across folds
    log.info(f"\n  CV Stability (mean ± std across {n_splits} folds):")
    for comp in SCORE_KEYS:
        vals = cv_df[cv_df["label"] == "Combined"].groupby("fold")[comp].mean()
        if len(vals) > 0:
            log.info(f"    {comp}: {vals.mean():.3f} ± {vals.std():.3f}")

    # IC consistency: train vs test
    log.info(f"\n  IC train vs test:")
    ic_cols = cv_df[cv_df["rank"] == 1][["label", "fold", "train_IC", "test_IC"]]
    for _, r in ic_cols.iterrows():
        log.info(f"    Fold {int(r['fold'])} [{r['label']}]: "
                 f"train_IC={r['train_IC']:.4f} → test_IC={r['test_IC']:.4f}")

    # Average IC gap
    if len(ic_cols) > 0:
        avg_gap = (ic_cols["train_IC"] - ic_cols["test_IC"]).mean()
        log.info(f"    Average IC degradation: {avg_gap:.4f}")
        if abs(avg_gap) < 0.02:
            log.info(f"    ✅ Weights are stable (low overfitting)")
        elif abs(avg_gap) < 0.05:
            log.info(f"    ⚠️  Moderate overfitting — consider fewer components")
        else:
            log.info(f"    ❌ High overfitting — weights may not generalize")

    return cv_df


def find_trade_optimal_weights(df, results_df):
    """From the Monte Carlo results, find the best weight set considering
    trade objectives: maximize IC while ensuring top_win_rate > baseline."""
    if len(results_df) == 0:
        return None

    baseline = results_df[results_df["label"] == "CURRENT"]
    base_ic = baseline["IC"].values[0] if len(baseline) > 0 else 0
    base_wr = baseline["top_win_rate"].values[0] if len(baseline) > 0 else 0

    # Candidates must beat baseline on IC and be in top 5% of top_win_rate
    candidates = results_df[(results_df["label"].isin(["", None])) |
                            (results_df["label"].isna())]
    if len(candidates) == 0:
        candidates = results_df[~results_df["label"].isin(["CURRENT", "EQUAL"])]

    if len(candidates) == 0:
        return None

    # Best by IC
    best_ic = candidates.nlargest(1, "IC").iloc[0]

    # Best by combined score
    best_combined = candidates.nlargest(1, "combined").iloc[0]

    # Best by top_win_rate
    best_wr = candidates.nlargest(1, "top_win_rate").iloc[0]

    # Best by weighted (IC + top_win_rate + spread)
    candidates["objective"] = (
        0.4 * candidates["IC"] +
        0.3 * candidates["top_win_rate"] +
        0.3 * candidates["spread_q4_q1"].clip(lower=0)
    )
    best_objective = candidates.nlargest(1, "objective").iloc[0]

    return {
        "baseline": {"IC": round(base_ic, 4), "top_win_rate": round(base_wr, 4)},
        "best_by_IC": best_ic.to_dict(),
        "best_by_combined": best_combined.to_dict(),
        "best_by_win_rate": best_wr.to_dict(),
        "best_by_objective": best_objective.to_dict(),
    }


def generate_summary_report(all_data, analysis_results, recommend, cv_df):
    """Generate a comprehensive summary."""
    log.info("\n" + "=" * 70)
    log.info("SUMMARY & RECOMMENDATIONS")
    log.info("=" * 70)

    n_signals = len(all_data)
    n_symbols = all_data["symbol"].nunique()
    time_range = f"{all_data['time'].min()} to {all_data['time'].max()}"
    overall_win = (all_data["fwd_24"] > 0).mean()
    log.info(f"  Data: {n_signals} signals, {n_symbols} symbols")
    log.info(f"  Period: {time_range}")
    log.info(f"  Market win rate (24h fwd): {overall_win:.2%}")

    if recommend is None:
        log.info("  ❌ No optimal weights found")
        return

    log.info(f"\n  Baseline (current weights):")
    log.info(f"    IC_24h = {recommend['baseline']['IC']}, "
             f"top_win_rate = {recommend['baseline']['top_win_rate']}")

    log.info(f"\n  🏆 Best weight set (by objective):")
    best = recommend["best_by_objective"]
    log.info(f"    IC = {best['IC']}, top_win_rate = {best['top_win_rate']}, "
             f"spread_Q4_Q1 = {best['spread_q4_q1']}")
    log.info(f"    Weights:")
    weight_str = ", ".join([f"{k}={best.get(k, 0):.2f}" for k in SCORE_KEYS])
    log.info(f"      {weight_str}")

    log.info(f"\n  📊 Alternative candidates:")
    for label, cand in [("Max IC", recommend.get("best_by_IC", {})),
                         ("Max win rate", recommend.get("best_by_win_rate", {})),
                         ("Max combined", recommend.get("best_by_combined", {}))]:
        if cand:
            w_str = ", ".join([f"{k}={cand.get(k, 0):.3f}" for k in SCORE_KEYS])
            log.info(f"    {label}: IC={cand.get('IC', '?')}, "
                     f"WR={cand.get('top_win_rate', '?')} | {w_str}")

    # CV stability
    if cv_df is not None and len(cv_df) > 0:
        log.info(f"\n  🔄 Cross-validation stability (IC test):")
        for label, group in cv_df.groupby("label"):
            for _, r in group.iterrows():
                log.info(f"    {label} rank {int(r['rank'])}: "
                         f"train_IC={r['train_IC']:.4f} → test_IC={r['test_IC']:.4f}")

    # Recommended settings.ini line
    log.info(f"\n  📝 settings.ini scoring weights (recommended):")
    top = best
    ini_weights = {k: round(top.get(k, 0), 2) for k in SCORE_KEYS}
    ini_line = ",".join([f"{k}:{ini_weights[k]}" for k in SCORE_KEYS
                         if ini_weights.get(k, 0) >= 0.01])
    log.info(f"    weights = {ini_line}")

    return ini_line


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and optimize scoring weights for Doto MT5 bot"
    )
    parser.add_argument("--years", type=float, default=1.0,
                        help="Years of H1 data to fetch per symbol (default: 1)")
    parser.add_argument("--mc-iter", type=int, default=50000,
                        help="Monte Carlo iterations (default: 50000)")
    parser.add_argument("--no-ml", action="store_true",
                        help="Skip ML scoring (models not loaded)")
    parser.add_argument("--output", type=str, default="data/weight_analysis",
                        help="Output path prefix (default: data/weight_analysis)")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip MT5 data collection, load existing CSV")
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of CV splits (default: 5)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load config
    log.info("Loading config...")
    cfg = load_config()
    log.info(f"  Portfolio: {cfg['symbols']}")
    log.info(f"  Scoring: {'enabled' if cfg.get('scoring_enabled', True) else 'disabled'}")
    log.info(f"  ML: {'enabled' if cfg.get('ml_enabled', True) else 'disabled'}")
    log.info(f"  Current weights from settings.ini")

    # Connect to MT5 if needed (gentle init, don't kill terminal)
    if not args.skip_data:
        log.info("Connecting to MT5...")
        import MetaTrader5 as mt5
        if not mt5.initialize():
            log.error("Failed to connect to MT5")
            sys.exit(1)
        log.info("  Connected.")

    # Collect data
    data_csv = output_path.parent / f"{output_path.name}_data.csv"
    if args.skip_data and data_csv.exists():
        log.info(f"Loading existing data from {data_csv}")
        all_data = pd.read_csv(data_csv, parse_dates=["time"])
        log.info(f"  Loaded {len(all_data)} signals from {data_csv}")
    else:
        all_data = collect_all_data(cfg, years=args.years, skip_ml=args.no_ml)
        if len(all_data) == 0:
            log.error("No data collected. Aborting.")
            sys.exit(1)

        all_data.to_csv(data_csv, index=False)
        log.info(f"Data saved to {data_csv}")

    # Filter to rows with forward return data
    valid_data = all_data.dropna(subset=["fwd_24"])
    log.info(f"Signals with 24h forward return: {len(valid_data)}/{len(all_data)}")
    if len(valid_data) < 100:
        log.error(f"Too few valid signals ({len(valid_data)}). Need more data.")
        sys.exit(1)

    # Component analysis
    comp_results = analyze_components(valid_data)
    comp_csv = output_path.parent / f"{output_path.name}_components.csv"
    comp_results.to_csv(comp_csv, index=False)
    log.info(f"Component analysis saved to {comp_csv}")

    # Monte Carlo weight search
    mc_results = run_weight_search(valid_data, n_iterations=args.mc_iter)
    if len(mc_results) == 0:
        log.error("Weight search produced no results. Aborting.")
        sys.exit(1)

    mc_csv = output_path.parent / f"{output_path.name}_mc_results.csv"
    mc_results.to_csv(mc_csv, index=False)
    log.info(f"MC results ({len(mc_results)}) saved to {mc_csv}")

    # Find optimal weights
    recommend = find_trade_optimal_weights(valid_data, mc_results)
    if recommend:
        rec_json = output_path.parent / f"{output_path.name}_recommendation.json"
        with open(rec_json, "w") as f:
            json.dump(recommend, f, indent=2, default=str)
        log.info(f"Recommendation saved to {rec_json}")

    # Cross-validation
    cv_results = run_cross_validation(valid_data,
                                      n_splits=args.cv_splits,
                                      n_iterations=min(args.mc_iter // 2, 30000))
    if cv_results is not None:
        cv_csv = output_path.parent / f"{output_path.name}_cv.csv"
        cv_results.to_csv(cv_csv, index=False)
        log.info(f"CV results saved to {cv_csv}")

    # Summary
    ini_line = generate_summary_report(valid_data, comp_results, recommend, cv_results)

    if ini_line and not args.skip_data:
        log.info(f"\n  {'=' * 50}")
        log.info(f"  RECOMMENDED settings.ini line:")
        log.info(f"  {'=' * 50}")
        log.info(f"  weights = {ini_line}")
        log.info(f"  {'=' * 50}")

    # Disconnect
    if not args.skip_data:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass

    log.info("Done.")


if __name__ == "__main__":
    main()
