"""Filter functions — volume, volatility, spread, tape, news, ML, tail risk, capital, session, daily loss."""

import logging
import time
from datetime import datetime, timedelta, timezone

import joblib
try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
import state as _st
from analytics import compute_entry_score, volume_filter_pass
from ml_features import FEATURE_COLS, prepare_features
from mt5_connect import get_rates, mt5_call
from regime import get_current_atr
from state import (
    ASSET_CLASS_MAP,
    BASE_DIR,
    _ml_meta_models,
    _ml_models,
    _ml_pool_models,
    _tail_risk_cooldown,
    _tail_risk_triggered,
)
from train_model import EnsembleModel


def check_volume_filter(cfg, signal):
    sym = cfg["symbol"]
    if not cfg.get("vf_enabled", True):
        return True
    needed = cfg.get("vf_sma_period", 20) + 5
    df = get_rates(sym, cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        logging.debug(f"[{sym}] Volume filter: insufficient data")
        return True
    # Delegate to the shared analytics module (single source of truth with the
    # backtest). It operates on the last CLOSED bar, so drop the forming bar.
    df_closed = df.iloc[:-1] if len(df) > 1 else df
    passed = volume_filter_pass(df_closed, signal, cfg)
    if not passed:
        logging.debug(f"[{sym}] Volume filter: rejected (rel_vol/OBV did not pass)")
    return passed


def check_ml_gate(cfg, signal, entry_atr):
    """Gate 3 — ML Validation: merged ML signal + scoring + news into pass/fail.

    Returns (passed: bool, confidence_mult: float, ml_conf: float).
    """
    if not cfg.get("ml_enabled", True) and not cfg.get("scoring_enabled", True):
        return True, 1.0, None

    scoring = cfg.get("scoring_enabled", True)
    if scoring:
        entry_score, score_details, ml_conf = compute_entry_score(cfg, signal, entry_atr)
        mr_min = 0.03 if entry_atr is None else 0.0
        min_score = cfg.get("scoring_min_entry", 0.60) + mr_min
        if entry_score < min_score:
            return False, 0.0, ml_conf
        high_bucket = cfg.get("scoring_confidence_bucket_high", 0.85)
        low_bucket = cfg.get("scoring_confidence_bucket_low", 0.60)
        if entry_score >= high_bucket:
            confidence_mult = cfg.get("scoring_high_conviction_mult", 1.0)
        elif entry_score >= low_bucket:
            confidence_mult = cfg.get("scoring_standard_edge_mult", 0.85)
        else:
            confidence_mult = cfg.get("scoring_low_conviction_mult", 0.50)
        news_val = score_details.get("news", 0.5) if score_details else 0.5
        if news_val >= 0.70:
            confidence_mult = min(1.5, confidence_mult * 1.10)
        elif news_val <= 0.30:
            confidence_mult *= 0.50
        return True, confidence_mult, ml_conf
    else:
        ml_pass, ml_conf = check_ml_signal(cfg, signal)
        if not ml_pass:
            return False, 0.0, ml_conf
        return True, 1.0, ml_conf


def check_execution_sanity(cfg, signal):
    """Gate 4 — Execution Sanity: merged volume + spread + tape.

    Returns bool (True = pass).
    """
    if not check_volume_filter(cfg, signal):
        return False
    if not check_spread_filter(cfg):
        return False
    return check_tape_reading(cfg, signal)


def check_spread_filter(cfg):
    if not cfg["spf_enabled"]:
        return True
    sym = cfg["symbol"]
    tick = mt5_call(mt5.symbol_info_tick, sym, _timeout=5)
    if tick is None:
        logging.debug(f"[{sym}] Spread filter: no tick")
        return True
    spread = tick.ask - tick.bid
    atr = get_current_atr(cfg)
    if atr is None or atr <= 0:
        logging.debug(f"[{sym}] Spread filter: no ATR")
        return True
    ratio = spread / atr
    if ratio > cfg["spf_max_ratio"]:
        logging.debug(f"[{sym}] Spread filter: spread/ATR {ratio:.3f} > {cfg['spf_max_ratio']}")
        return False
    return True


def check_tape_reading(cfg, signal):
    if not cfg["tape_enabled"]:
        return True
    sym = cfg["symbol"]
    lookback = cfg["tape_m1_lookback"]
    rates = mt5_call(mt5.copy_rates_from_pos, sym, mt5.TIMEFRAME_M1, 0, lookback, _timeout=5)
    if rates is None or len(rates) < 10:
        logging.debug(f"[{sym}] Tape reading: insufficient M1 data")
        return True
    df = pd.DataFrame(rates)
    avg_range = (df["high"] - df["low"]).mean()
    df["bullish_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
    df["range_ratio"] = (df["high"] - df["low"]) / (avg_range + 1e-10)
    recent = df.tail(5)
    avg_pressure = recent["bullish_pressure"].mean()
    range_active = recent["range_ratio"].mean()
    if (
        signal == "buy"
        and avg_pressure < cfg.get("tape_bearish_pressure", 0.35)
        and range_active > cfg.get("tape_range_ratio", 1.2)
    ):
        logging.debug(f"[{sym}] Tape: low bullish pressure {avg_pressure:.3f} blocking buy")
        return False
    if (
        signal == "sell"
        and avg_pressure > cfg.get("tape_bullish_pressure", 0.65)
        and range_active > cfg.get("tape_range_ratio", 1.2)
    ):
        logging.debug(f"[{sym}] Tape: high bullish pressure {avg_pressure:.3f} blocking sell")
        return False
    return True


def check_ml_signal(cfg, signal, df=None):
    if not cfg["ml_enabled"]:
        return True, None
    symbol = cfg["symbol"]
    model_entry = _ml_models.get(symbol)
    if model_entry is None:
        asset_class = ASSET_CLASS_MAP.get(symbol)
        if asset_class:
            pool_entry = _ml_pool_models.get(asset_class)
            if pool_entry is not None:
                model_entry = pool_entry
    if model_entry is None:
        return True, None
    model = model_entry["model"]
    metadata = model_entry.get("metadata", {})
    model_features = metadata.get("features", FEATURE_COLS)
    model_type = metadata.get("model_type", "ensemble")
    needed = 250
    if df is None:
        df = get_rates(symbol, cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        return True, None
    feat_result = prepare_features(df, symbol=symbol)
    if feat_result is None:
        return True, None
    feature_data, full_df = feat_result
    missing = [c for c in model_features if c not in feature_data.columns]
    if missing:
        logging.debug(f"[{symbol}] ML: missing features {missing} — pass-through")
        return True, None
    latest = feature_data[model_features].iloc[-1:]
    latest_X = latest.fillna(0)
    latest_arr = np.nan_to_num(latest_X.values, nan=0.0)
    if latest_arr.shape[0] == 0:
        return True, None
    n_expected = None
    try:
        if hasattr(model, "n_features_in_"):
            n_expected = model.n_features_in_
        elif hasattr(model, "xgb") and hasattr(model.xgb, "n_features_in_"):
            n_expected = model.xgb.n_features_in_
        elif hasattr(model, "lgb") and hasattr(model.lgb, "n_features_in_"):
            n_expected = model.lgb.n_features_in_
    except Exception:
        logging.debug("[%s] Could not determine ML n_features_in_", symbol)
    if n_expected is not None and latest_arr.shape[1] != n_expected:
        logging.warning(
            f"[{symbol}] ML feature count mismatch: data has {latest_arr.shape[1]}, "
            f"model expects {n_expected} — pass-through"
        )
        return True, None

    # Dispatch by model type
    if model_type == "regressor":
        expected_r = float(model.predict(latest_arr)[0])
        min_r = cfg.get("ml_min_r", 0.20)
        feature_stats = metadata.get("feature_stats")
        if feature_stats and len(feature_stats) > 0:
            _check_feature_drift(symbol, feature_data, feature_stats)
        if expected_r < min_r:
            logging.debug(f"[{symbol}] ML regressor: expected R {expected_r:.3f} < min {min_r:.3f}")
            return False, float(expected_r)
        return True, float(expected_r)
    else:
        # If the ensemble carries a TFT member it was trained on seq_len-bar
        # windows; feeding it a single row (latest_arr) strips all temporal
        # context and produces off-distribution predictions (agent audit D2).
        # predict_proba returns one row per input row, and XGB/LGB are per-row
        # independent, so we pass the last seq_len rows and read the last row:
        # the TFT then sees a full window ending at the latest bar.
        tft_member = getattr(model, "tft", None)
        if tft_member is not None:
            seq_len = getattr(tft_member, "seq_len", 20)
            X = feature_data[model_features].iloc[-seq_len:].fillna(0)
        else:
            X = latest_X
        proba = model.predict_proba(X)
        prob_long = proba[-1][1] if proba.shape[1] > 1 else proba[-1][0]
        conf = prob_long if signal == "buy" else 1.0 - prob_long
        opt_threshold = metadata.get("optimal_threshold")
        threshold = cfg["ml_threshold_overrides"].get(symbol)
        if threshold is None:
            threshold = opt_threshold if opt_threshold is not None else cfg["ml_confidence"]
        if (
            opt_threshold is not None
            and threshold == opt_threshold
            and conf < threshold
            and conf >= cfg["ml_confidence"]
        ):
            threshold = cfg["ml_confidence"]
        if conf < threshold:
            logging.debug(f"[{symbol}] ML: conf {conf:.3f} < threshold {threshold:.3f}")
            return False, conf
        meta_entry = _ml_meta_models.get(symbol)
        if meta_entry is not None:
            try:
                meta_model = meta_entry["model"]
                meta_features = meta_entry.get("metadata", {}).get("features", model_features)
                meta_latest = feature_data[meta_features].iloc[-1:]
                meta_arr = np.nan_to_num(meta_latest.values.copy(), nan=0.0)
                meta_proba = meta_model.predict_proba(meta_arr)
                meta_conf = meta_proba[0][1] if meta_proba.shape[1] > 1 else meta_proba[0][0]
                meta_threshold = cfg.get("ml_meta_threshold", 0.50)
                if meta_conf < meta_threshold:
                    logging.debug(f"[{symbol}] ML meta: conf {meta_conf:.3f} < threshold {meta_threshold:.3f}")
                    return False, conf * meta_conf
            except Exception as e:
                logging.warning(f"[{symbol}] Meta-labeler inference failed: {e}")
        if symbol not in _st._ml_confidence_history:
            _st._ml_confidence_history[symbol] = []
        hist = _st._ml_confidence_history[symbol]
        hist.append(conf)
        if len(hist) > 200:
            _st._ml_confidence_history[symbol] = hist[-200:]
        if len(hist) >= 50 and len(hist) % 20 == 0:
            baseline = _st._ml_confidence_baseline.get(symbol)
            if baseline is None:
                baseline = float(np.mean(hist[:50]))
                _st._ml_confidence_baseline[symbol] = baseline
            recent = np.mean(hist[-50:])
            if baseline > 0.3 and recent < baseline * 0.85 and symbol not in _st._ml_drift_warned:
                _st._ml_drift_warned.add(symbol)
                logging.warning(
                    f"[{symbol}] ML model drift detected: baseline conf {baseline:.2f} → "
                    f"recent {recent:.2f} ({(1 - recent / baseline) * 100:.0f}% drop)"
                )
        feature_stats = metadata.get("feature_stats")
        if feature_stats and len(feature_stats) > 0:
            _check_feature_drift(symbol, feature_data, feature_stats)
        return True, conf


def _check_feature_drift(symbol, feature_data, feature_stats):
    """Check per-feature drift vs training distribution. Warns if >3 features drift >2σ."""
    drifted = []
    for col, stats in feature_stats.items():
        if col not in feature_data.columns:
            continue
        mu = stats.get("mean")
        sigma = stats.get("std")
        if mu is None or sigma is None or sigma <= 0:
            continue
        cur = feature_data[col].iloc[-1]
        if np.isnan(cur):
            continue
        z = abs(cur - mu) / sigma
        if z > 2.0:
            drifted.append((col, z))
    if len(drifted) >= 3:
        top = ", ".join(f"{c}(z={z:.1f})" for c, z in drifted[:5])
        logging.warning(f"[{symbol}] Feature drift: {len(drifted)} features >2σ — retraining recommended. {top}")
        if symbol not in _st._ml_drift_warned:
            _st._ml_drift_warned.add(symbol)
            logging.warning(f"[{symbol}] First drift alert — scheduling warm-start")
            try:
                from drift_detector import schedule_warmstart

                schedule_warmstart(symbol)
            except Exception:
                logging.debug("[%s] Could not schedule warmstart", symbol)


def check_tail_risk(cfg):
    if not cfg["tr_enabled"]:
        return True
    sym = cfg["symbol"]
    now = time.time()
    if _tail_risk_triggered.get(sym, False) and now >= _tail_risk_cooldown.get(sym, 0):
        _st._tail_risk_triggered.pop(sym, None)
    needed = cfg["tr_lookback"] + 10
    df = get_rates(sym, cfg["timeframe"], needed)
    if df is not None and len(df) >= needed:
        returns = df["close"].pct_change().dropna().values[-cfg["tr_lookback"] :]
        if len(returns) > 10:
            mean_r = np.mean(returns)
            std_r = np.std(returns)
            last_r = returns[-1]
            if std_r > 0 and abs(last_r - mean_r) / std_r > cfg["tr_sigma"]:
                logging.info(f"[{sym}] Tail risk: {cfg['tr_sigma']}σ event detected")
                _st._tail_risk_triggered[sym] = True
                _st._tail_risk_cooldown[sym] = now + cfg["tr_cooldown"] * 60
                return False
    acc = mt5_call(mt5.account_info, _timeout=5)
    if acc is not None:
        equity = acc.balance + acc.profit
        if equity > _st._peak_balance:
            _st._peak_balance = equity
        peak = max(_st._peak_balance, 1)
        dd_pct = ((peak - equity) / peak) * 100
        # NOTE: the circuit-breaker (15%) check MUST come before the tail-risk
        # (8%) early-return, otherwise the 15% permanent halt is unreachable
        # (agent audit C2).
        if cfg.get("cb_enabled", True) and dd_pct >= cfg.get("cb_dd_pct", 15.0):
            logging.critical(f"[{sym}] Circuit breaker: drawdown {dd_pct:.1f}% >= {cfg.get('cb_dd_pct', 15.0)}%")
            _st._circuit_breaker_triggered = True
            return False
        if dd_pct >= cfg["tr_max_dd_pct"]:
            logging.info(f"[{sym}] Tail risk: drawdown {dd_pct:.1f}% >= {cfg['tr_max_dd_pct']}%")
            _st._tail_risk_triggered[sym] = True
            _st._tail_risk_cooldown[sym] = now + cfg["tr_cooldown"] * 60
            return False
    return True


def check_capital_eligibility(cfg, symbol):
    min_equity = cfg["symbol_strategy"].get(symbol, {}).get("min_equity")
    if min_equity is None or min_equity <= 0:
        return True
    acc = mt5_call(mt5.account_info, _timeout=5)
    if acc is None:
        logging.debug(f"[{symbol}] Capital eligibility: no account info")
        return True
    if acc.equity < min_equity:
        logging.debug(f"[{symbol}] Capital eligibility: equity {acc.equity} < {min_equity}")
        return False
    return True


def _mt5_daily_realized_loss(today):
    """Authoritative realized loss for today from MT5 deal history.

    Sums net (profit + commission + swap) over all deals with time >= start of
    the bot's trading day. This captures EVERY closed trade -- bot, manual, and
    external -- so the daily-loss halt can no longer be blinded by a trade the
    bot didn't place (the 20k manual BTCUSD loss case). Returns None if MT5
    deal history is unavailable, so callers fall back to the journal counter.
    """
    try:
        from datetime import datetime as _dt

        pkt = timezone(timedelta(hours=5))
        start_dt = _dt.combine(today, _dt.min.time()).replace(tzinfo=pkt)
        start_ts = int(start_dt.timestamp())
        deals = mt5_call(mt5.history_deals_get, start_ts, int(_dt.now().timestamp()) + 10, _timeout=10)
        if deals is None:
            return None
        net = 0.0
        for d in deals:
            try:
                if d.time < start_ts:
                    continue
                # Skip balance/deposit operations (position_id == 0) so a
                # deposit does not mask or inflate realized PnL.
                if getattr(d, "position_id", 0) == 0:
                    continue
                net += (d.profit or 0.0) + (d.commission or 0.0) + (d.swap or 0.0)
            except Exception:
                logging.debug("Failed to process deal in _mt5_daily_realized_loss", exc_info=True)
                continue
        return net
    except Exception:
        logging.debug("_mt5_daily_realized_loss failed", exc_info=True)
        return None


_daily_loss_check_date = None


def check_daily_loss(cfg):
    pkt = timezone(timedelta(hours=5))
    today = datetime.now(pkt).date()
    global _daily_loss_check_date
    if _daily_loss_check_date != today:
        _daily_loss_check_date = today
        _st._daily_loss_hit = False
        _st._daily_loss_mt5_cache = (None, None, 0)
    if _st._daily_loss_hit:
        return False
    account_info = mt5_call(mt5.account_info)
    if account_info is None:
        return True
    balance = account_info.balance
    if balance <= 0:
        logging.critical(f"Zero/negative balance ({balance}) — account depleted")
        return False
    # MT5 deal history is authoritative (sees external/manual trades).
    # Journal counter (`_daily_realized_pnl`) is updated on every trade close
    # and captures intra-cycle losses the MT5 index may not have indexed yet.
    # Cache the MT5 result for 120s to avoid hammering MT5 7x per cycle.
    # Between refreshes, `_daily_realized_pnl` provides intra-cycle freshness.
    cached_day, cached_loss, cached_ts = getattr(_st, "_daily_loss_mt5_cache", (None, None, 0))
    mt5_loss = (
        cached_loss
        if (cached_day == today and cached_loss is not None and time.time() - cached_ts < 120)
        else _mt5_daily_realized_loss(today)
    )
    if mt5_loss is not None:
        if mt5_loss != cached_loss or cached_ts == 0:
            _st._daily_loss_mt5_cache = (today, mt5_loss, time.time())
        journal_pnl = _st._daily_realized_pnl if _st._daily_realized_date == today else 0.0
        if journal_pnl != 0 and mt5_loss != 0:
            drift_pct = abs(mt5_loss - journal_pnl) / max(abs(mt5_loss), 1)
            if drift_pct > 0.10:
                logging.info(
                    f"[daily_loss] MT5 loss={mt5_loss:.2f} diverges from journal loss={journal_pnl:.2f} "
                    f"({drift_pct * 100:.0f}% drift)"
                )
        realized_loss = min(mt5_loss, journal_pnl)
    else:
        journal_pnl = _st._daily_realized_pnl if _st._daily_realized_date == today else 0.0
        realized_loss = journal_pnl
    if realized_loss >= 0:
        return True
    loss_pct = (-realized_loss / balance) * 100
    if loss_pct >= cfg["daily_loss_pct"]:
        _st._daily_loss_hit = True
        logging.warning(
            f"Daily loss limit reached: realized {realized_loss:.2f} ({loss_pct:.2f}% of {balance:.2f}) "
            f"— halting entries"
        )
        return False
    return True


def load_ml_models(cfg):
    if not cfg["ml_enabled"]:
        return
    import __main__

    __main__.EnsembleModel = EnsembleModel
    from train_model import EnsembleRegressor as _ER

    __main__.EnsembleRegressor = _ER
    from train_model import _CalibratedWrapper

    __main__._CalibratedWrapper = _CalibratedWrapper
    for symbol in cfg["symbols"]:
        safe_symbol = symbol.replace(".", "_")
        model_path_str = cfg["ml_model_path_template"].format(symbol=safe_symbol)
        model_path = BASE_DIR / model_path_str
        if model_path.exists():
            try:
                data = joblib.load(model_path)
                _ml_models[symbol] = data
                logging.info(f"[{symbol}] ML model loaded from {model_path}")
            except Exception as e:
                logging.warning(f"[{symbol}] Failed to load ML model: {type(e).__name__}: {e}", exc_info=True)
        else:
            logging.warning(f"[{symbol}] ML model not found at {model_path}")
        meta_path = BASE_DIR / f"models/model_{safe_symbol}.meta.pkl"
        if meta_path.exists():
            try:
                meta_data = joblib.load(meta_path)
                _ml_meta_models[symbol] = meta_data
                logging.info(f"[{symbol}] Meta-labeler loaded from {meta_path}")
            except Exception as e:
                logging.warning(f"[{symbol}] Failed to load meta-labeler: {e}")
    for symbol in cfg["symbols"]:
        asset_class = ASSET_CLASS_MAP.get(symbol)
        if asset_class and asset_class not in _ml_pool_models:
            pool_path = BASE_DIR / f"models/model_pool_{asset_class}.pkl"
            if pool_path.exists():
                try:
                    data = joblib.load(pool_path)
                    _ml_pool_models[asset_class] = data
                    logging.info(f"[pool] {asset_class} model loaded from {pool_path}")
                except Exception as e:
                    _ml_pool_models[asset_class] = None
                    logging.warning(f"[pool] Failed to load {asset_class} model: {e}")
