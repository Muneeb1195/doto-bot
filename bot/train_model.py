import argparse
import configparser
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
import pandas as pd
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from ml_features import FEATURE_COLS, compute_feature_stats, prepare_features, r_multiple_labels, triple_barrier_labels

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

_settings = None


def _get_settings():
    global _settings
    if _settings is None:
        _settings = configparser.ConfigParser()
        _settings.read(CONFIG_DIR / "settings.ini")
    return _settings


ASSET_CLASS_MAP = {
    "ETHUSD.raw": "crypto",
    "BTCUSD.raw": "crypto",
    "LTCUSD.raw": "crypto",
    "DOGUSD.raw": "crypto",
    "ADAUSD.raw": "crypto",
    "BNBUSD.raw": "crypto",
    "XRPUSD.raw": "crypto",
    "SOLUSD.raw": "crypto",
    "EURUSD.raw": "forex",
    "GBPUSD.raw": "forex",
    "USDJPY.raw": "forex",
    "EURJPY.raw": "forex",
    "GBPJPY.raw": "forex",
    "NZDUSD.raw": "forex",
    "AUDUSD.raw": "forex",
    "USDCAD.raw": "forex",
    "USDCHF.raw": "forex",
    "XAUUSD.raw": "commodity",
    "XAGUSD.raw": "commodity",
    "XNGUSD.raw": "commodity",
    "XAU500.raw": "commodity",
    "XPTUSD.raw": "commodity",
    "US30.raw": "index",
    "SPY.raw": "index",
    "US500.raw": "index",
    "IWM.raw": "index",
}


class PurgedTimeSeriesSplit:
    def __init__(self, n_splits=5, gap=12):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):  # noqa: ARG002
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            start_test = (i + 1) * fold_size
            end_test = start_test + fold_size if i < self.n_splits - 1 else n_samples
            train_end = max(0, start_test - self.gap)
            yield np.arange(0, train_end), np.arange(start_test, end_test)


def _resolve_label_params(symbol):
    cfg = _get_settings()
    sym_section = f"STRATEGY:{symbol}"
    if cfg.has_section(sym_section):
        sl_mult = cfg.getfloat(sym_section, "atr_sl_multiplier", fallback=None)
        rr = cfg.getfloat(sym_section, "risk_reward_ratio", fallback=None)
        if sl_mult is not None and rr is not None:
            sl_atr = sl_mult
            tp_atr = sl_mult * rr
            return tp_atr, sl_atr
    sl_mult = cfg.getfloat("STRATEGY", "atr_sl_multiplier", fallback=1.0)
    rr = cfg.getfloat("STRATEGY", "risk_reward_ratio", fallback=2.0)
    return sl_mult * rr, sl_mult


def load_csv_data_train(symbol, tf_name="H1"):
    """Load pre-exported bars from data/history/<SYMBOL>_<TF>.csv.

    Returns a DataFrame (time as datetime) or None if CSV missing.
    """
    csv_path = BASE_DIR / "data" / "history" / f"{symbol.replace('.', '_')}_{tf_name}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        return None
    if pd.api.types.is_numeric_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], unit="s")
    else:
        df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    if "spread" not in df.columns:
        df["spread"] = 0
    return df


def fetch_data(symbol, years=3, tf="H1", csv_mode=False):
    if csv_mode:
        return load_csv_data_train(symbol, tf_name=tf)
    if not mt5.initialize():
        print(f"Failed to initialize MT5: {mt5.last_error()}")
        return None
    import MetaTrader5 as mt5_mod
    tf_const = getattr(mt5_mod, f"TIMEFRAME_{tf}", mt5_mod.TIMEFRAME_H1)
    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    rates = mt5_mod.copy_rates_range(symbol, tf_const, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    print(f"Fetched {len(df)} bars for {symbol} ({start.date()} to {end.date()})")
    return df


def fetch_m1_data(symbol, years=3, csv_mode=False):
    """Fetch M1 OHLCV+spread bars for historical orderflow feature backfill.

    Orderflow features are otherwise only computed live for the last bar, which
    leaves every historical training row as of_* = NaN -> 0.0 (train/serve skew).
    Supplying M1 bars lets prepare_features/compute_features fill real of_* for
    all rows. Uses backward paging (fetch_rates_paged) so per-request bar caps
    don't silently truncate the window. Returns None (graceful) if M1 is
    unavailable.
    """
    if csv_mode:
        return load_csv_data_train(symbol, tf_name="M1")
    if not mt5.initialize():
        return None
    from mt5_connect import fetch_rates_paged

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    return fetch_rates_paged(symbol, mt5.TIMEFRAME_M1, start, end)


def tune_xgb_params(X_train, y_train, sw, n_trials=30):
    import optuna
    import xgboost as xgb
    from sklearn.metrics import log_loss
    from sklearn.model_selection import TimeSeriesSplit

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": sw,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": 0,
        }
        tscv = TimeSeriesSplit(n_splits=3)
        losses = []
        for tr_idx, va_idx in tscv.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_va)
            losses.append(log_loss(y_va, proba))
        return sum(losses) / len(losses)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best params: {study.best_params}")
    print(f"  Best CV log_loss: {study.best_value:.4f}")

    tuned = study.best_params
    tuned["scale_pos_weight"] = sw
    tuned["objective"] = "binary:logistic"
    tuned["eval_metric"] = "logloss"
    tuned["random_state"] = 42
    tuned["n_jobs"] = 1
    tuned["verbosity"] = 0
    return tuned


LGB_FIXED_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "objective": "binary",
    "verbose": -1,
    "random_state": 42,
    "n_jobs": 1,
}


class EnsembleModel:
    def __init__(self, xgb_model, lgb_model, tft_model=None):
        self.xgb = xgb_model
        self.lgb = lgb_model
        self.tft = tft_model

    def __getattr__(self, name):
        if name == "tft":
            return None
        raise AttributeError(name)

    def predict(self, X):
        xgb_pred = self.xgb.predict(X)
        lgb_pred = self.lgb.predict(X)
        n_models = 2
        tft_model = self.tft
        if tft_model is not None:
            tft_pred = tft_model.predict(X)
            n_models = 3
            avg = (xgb_pred.astype(float) + lgb_pred.astype(float) + tft_pred.astype(float)) / n_models
        else:
            avg = (xgb_pred.astype(float) + lgb_pred.astype(float)) / n_models
        return np.round(avg).astype(int)

    def predict_proba(self, X):
        xgb_proba = self.xgb.predict_proba(X)
        lgb_proba = self.lgb.predict_proba(X)
        n_models = 2
        tft_model = self.tft
        if tft_model is not None:
            tft_proba = tft_model.predict_proba(X)
            n_models = 3
            avg = (xgb_proba + lgb_proba + tft_proba) / n_models
        else:
            avg = (xgb_proba + lgb_proba) / n_models
        return avg

    @property
    def classes_(self):
        return self.xgb.classes_

    def get_params(self, deep=True):  # noqa: ARG002
        return {}


class EnsembleRegressor:
    def __init__(self, xgb_model, lgb_model):
        self.xgb = xgb_model
        self.lgb = lgb_model

    def predict(self, X):
        xgb_pred = self.xgb.predict(X)
        lgb_pred = self.lgb.predict(X)
        return (xgb_pred.astype(float) + lgb_pred.astype(float)) / 2

    def get_params(self, deep=True):  # noqa: ARG002
        return {}


FIXED_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
    "verbosity": 0,
}


def train_xgb(X_train, y_train, scale_pos_weight=None, tune_params=None):
    import xgboost as xgb

    if tune_params:
        params = tune_params.copy()
    else:
        params = FIXED_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight if scale_pos_weight else 1

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, np.array([0, 1])


LGB_PARAM_MAP = {
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "n_jobs",
}


def train_lgb(X_train, y_train, scale_pos_weight=None, tune_params=None):
    import lightgbm as lgb

    params = {k: v for k, v in tune_params.items() if k in LGB_PARAM_MAP} if tune_params else LGB_FIXED_PARAMS.copy()
    if scale_pos_weight and scale_pos_weight > 1:
        params["scale_pos_weight"] = scale_pos_weight

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model


def optimize_threshold(y_true, y_proba, rr=2.0, min_trades=60):
    best_ev = -1e9
    best_threshold = 0.5
    for t in np.arange(0.30, 0.86, 0.01):
        y_pred = (y_proba[:, 1] >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        n_trades = tp + fp
        if n_trades < min_trades:
            continue
        ev = tp * rr - fp
        if ev > best_ev:
            best_ev = ev
            best_threshold = t
    if best_ev == -1e9:
        print(f"  WARNING: no threshold met min_trades={min_trades}, returning default 0.5")
    n_trades_at = int((y_proba[:, 1] >= best_threshold).sum())
    print(f"  Optimal threshold: {best_threshold:.2f} (ev={best_ev:.0f}, trades={n_trades_at})")
    return best_threshold


def train_fold_ensemble(X_tr, y_tr, X_te, y_te, sw, tune_params=None, use_tft=False):
    # Hold out a calibration set so the isotonic calibrator is NEVER fit on the
    # evaluation set (fixes CR-BT2 calibration leak). Calibration is carved from
    # the TRAINING fold by default. Only when the training fold is too small do
    # we borrow from the test fold — and even then we cap calibration at half
    # the test fold and require a usable eval remainder, so calibration data and
    # evaluation data are always disjoint (no test-set leak).
    n_tr = len(X_tr)
    n_te = len(X_te)
    n_cal = min(max(n_tr // 10, 50), 200)
    if n_cal < n_tr:
        # Healthy case: calibration ⊂ training fold, disjoint from test fold.
        X_cal, y_cal = X_tr[-n_cal:], y_tr[-n_cal:]
        X_tr_final, y_tr_final = X_tr[:-n_cal], y_tr[:-n_cal]
    elif n_te >= 100:
        # Training too small — borrow from test, but cap at half so the other
        # half remains a clean, disjoint evaluation set.
        n_cal = min(max(n_te // 2, 50), n_te // 2)
        X_cal, y_cal = X_te[:n_cal], y_te[:n_cal]
        X_te, y_te = X_te[n_cal:], y_te[n_cal:]
        X_tr_final, y_tr_final = X_tr, y_tr
    else:
        # Neither fold is large enough to hold out clean calibration — train and
        # evaluate on the full folds with NO isotonic calibration (avoid leaking
        # the eval set into the calibrator entirely).
        X_cal, y_cal = None, None
        X_tr_final, y_tr_final = X_tr, y_tr
    xgb_model, _ = train_xgb(X_tr_final, y_tr_final, scale_pos_weight=sw, tune_params=tune_params)
    lgb_model = train_lgb(X_tr_final, y_tr_final, scale_pos_weight=sw, tune_params=tune_params)
    tft_model = None
    if use_tft and len(X_tr_final) >= 100:
        try:
            from tft_model import train_tft

            X_val_flat = X_te if len(X_te) >= 50 else None
            y_val_flat = y_te if len(X_te) >= 50 else None
            tft_model = train_tft(X_tr_final, y_tr_final, X_val=X_val_flat, y_val=y_val_flat)
        except Exception as e:
            print(f"  TFT training skipped: {e}")
    raw_importances = xgb_model.feature_importances_.copy() if hasattr(xgb_model, "feature_importances_") else None
    xgb_cal = calibrate_model(xgb_model, X_cal, y_cal)
    lgb_cal = calibrate_model(lgb_model, X_cal, y_cal)
    if tft_model is not None:
        try:
            tft_cal_n = min(n_cal, len(X_tr_final) // 5)
            if tft_cal_n >= 30:
                from sklearn.isotonic import IsotonicRegression

                raw = tft_model.predict_proba(X_tr_final[:tft_cal_n])[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_tr_final[:tft_cal_n])
                tft_model = _CalibratedWrapper(tft_model, iso)
        except Exception:
            tft_model = None
    ensemble = EnsembleModel(xgb_cal, lgb_cal, tft_model=tft_model)
    y_pred = ensemble.predict(X_te)
    y_proba = ensemble.predict_proba(X_te)
    score = f1_score(y_te, y_pred)
    return ensemble, score, xgb_cal, lgb_cal, raw_importances, y_proba, y_te


class _CalibratedWrapper:
    """Wraps a model with isotonic calibration (sklearn 1.9+ compat)."""

    def __init__(self, base_model, isotonic_reg):
        self.base_model = base_model
        self.isotonic_reg = isotonic_reg

    def predict(self, X):
        raw = self.base_model.predict(X)
        return raw

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)
        cal = self.isotonic_reg.predict(raw[:, 1])
        cal = np.clip(cal, 0, 1)
        out = np.zeros_like(raw)
        out[:, 0] = 1 - cal
        out[:, 1] = cal
        return out

    def get_feature_importances(self):
        if hasattr(self.base_model, "feature_importances_"):
            return self.base_model.feature_importances_
        return None

    @property
    def feature_importances_(self):
        imp = self.get_feature_importances()
        if imp is not None:
            return imp
        raise AttributeError("base model has no feature_importances_")

    @property
    def classes_(self):
        return self.base_model.classes_


def calibrate_model(model, X_calib, y_calib, method="isotonic"):  # noqa: ARG001
    if X_calib is None or len(X_calib) < 50:
        return model
    from sklearn.isotonic import IsotonicRegression

    try:
        # Ensemble models store their base estimators as `xgb` / `lgb`.
        # Calibrate each base model individually so the ensemble average
        # reflects calibrated member probabilities (not a double-wrapped average).
        if hasattr(model, "xgb") and hasattr(model, "lgb"):
            xgb_raw = model.xgb.predict_proba(X_calib)[:, 1]
            iso_xgb = IsotonicRegression(out_of_bounds="clip").fit(xgb_raw, y_calib)
            model.xgb = _CalibratedWrapper(model.xgb, iso_xgb)

            lgb_raw = model.lgb.predict_proba(X_calib)[:, 1]
            iso_lgb = IsotonicRegression(out_of_bounds="clip").fit(lgb_raw, y_calib)
            model.lgb = _CalibratedWrapper(model.lgb, iso_lgb)
            logging.info(f"Calibrated ensemble (isotonic) on {len(X_calib)} holdout samples")
        else:
            raw = model.predict_proba(X_calib)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_calib)
            model = _CalibratedWrapper(model, iso)
            logging.info(f"Calibrated model (isotonic) on {len(X_calib)} holdout samples")
    except Exception as e:
        logging.warning(f"Calibration failed: {e}")
    return model


def prune_features(feat_names, importances, keep_ratio=0.7):
    if importances.sum() == 0:
        print("  WARNING: all importances zero, skipping pruning")
        return feat_names
    sorted_idx = np.argsort(importances)[::-1]
    n_keep = max(1, int(len(feat_names) * keep_ratio))
    kept = [feat_names[i] for i in sorted_idx[:n_keep]]
    dropped = [feat_names[i] for i in sorted_idx[n_keep:]]
    if dropped:
        print(f"  Pruned {len(dropped)} features: {', '.join(dropped[:5])}{'...' if len(dropped) > 5 else ''}")
    return kept


def train_xgb_regressor(X, y, tune_params=None):
    import xgboost as xgb

    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "objective": "reg:absoluteerror",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if tune_params:
        params.update(
            {k: v for k, v in tune_params.items() if k in params or k in ("n_estimators", "max_depth", "learning_rate")}
        )
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def train_lgb_regressor(X, y):
    import lightgbm as lgb

    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": 20,
        "objective": "regression_l1",
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    return model


def train_regression_ensemble(X_tr, y_tr, X_te, y_te, tune_params=None):
    X_tr_final, y_tr_final = X_tr, y_tr
    xgb_model = train_xgb_regressor(X_tr_final, y_tr_final, tune_params=tune_params)
    lgb_model = train_lgb_regressor(X_tr_final, y_tr_final)
    ensemble = EnsembleRegressor(xgb_model, lgb_model)
    from sklearn.metrics import mean_absolute_error

    y_pred = ensemble.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    return ensemble, mae, xgb_model, lgb_model


def train_meta_labeler(X, y_primary, y_true, tune_params=None):
    neg = (y_primary == y_true).sum()
    pos = (y_primary != y_true).sum()
    correct = (y_primary == y_true).astype(int)
    if pos == 0 or neg == 0:
        return None
    from sklearn.model_selection import train_test_split

    X_m_tr, X_m_te, y_m_tr, y_m_te = train_test_split(np.nan_to_num(X, nan=0.0), correct, test_size=0.2, shuffle=False)
    sw = neg / pos
    params = (
        tune_params.copy()
        if tune_params
        else {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "scale_pos_weight": sw,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 1,
            "verbosity": 0,
        }
    )
    import xgboost as xgb

    meta_model = xgb.XGBClassifier(**params)
    meta_model.fit(X_m_tr, y_m_tr)
    meta_proba = meta_model.predict_proba(X_m_te)
    meta_score = f1_score(y_m_te, (meta_proba[:, 1] >= 0.5).astype(int))
    print(f"  Meta-labeler: f1={meta_score:.3f}, correct_rate={correct.mean():.3f}")
    return meta_model


def train_pool_model(symbols, years=3, tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold=12, tuned_params=None, tft=False):
    class_label = None
    for sym in symbols:
        if sym in ASSET_CLASS_MAP:
            class_label = ASSET_CLASS_MAP[sym]
            break
    if not class_label:
        print("No common asset class found")
        return None

    sym_tp, sym_sl = _resolve_label_params(symbols[0])
    tp_atr_mult = sym_tp
    sl_atr_mult = sym_sl
    print(f"  Pool triple-barrier: tp_atr={tp_atr_mult:.2f} sl_atr={sl_atr_mult:.2f} max_hold={max_hold}")
    print(f"\n{'=' * 60}")
    print(f"Training POOL model for {class_label.upper()} ({len(symbols)} symbols)")
    print(f"{'=' * 60}")
    all_X = []
    all_y = []
    for symbol in symbols:
        df = fetch_data(symbol, years)
        if df is None or len(df) < 200:
            continue
        df_m1 = fetch_m1_data(symbol, years)
        feature_data, full_df = prepare_features(df, symbol=symbol, m1_df=df_m1)
        labels = triple_barrier_labels(full_df, tp_atr_mult, sl_atr_mult, max_hold)
        aligned = pd.concat([feature_data, labels], axis=1).dropna(subset=["label"])
        aligned = aligned[aligned.index.isin(feature_data.index)]
        X_sym = aligned[FEATURE_COLS].values
        y_sym = aligned["label"].values
        mask = y_sym != 0
        all_X.append(X_sym[mask])
        all_y.append(((y_sym[mask] == 1).astype(np.int8)))
    if len(all_X) < 2:
        print("Insufficient pooled data")
        return None
    X_pool = np.vstack(all_X)
    y_pool = np.concatenate(all_y)
    X_pool = np.nan_to_num(X_pool, nan=0.0)
    # Hold out calibration samples (last in temporal order)
    n_calib = min(200, max(50, int(len(X_pool) * 0.15)))
    has_calib = n_calib >= 50 and len(X_pool) - n_calib >= 300
    if has_calib:
        X_calib = X_pool[-n_calib:]
        y_calib = y_pool[-n_calib:]
        X_pool = X_pool[:-n_calib]
        y_pool = y_pool[:-n_calib]
    else:
        X_calib = np.array([])
        y_calib = np.array([])
    n_splits = min(5, max(2, len(X_pool) // 100))
    pts = PurgedTimeSeriesSplit(n_splits=n_splits, gap=max_hold)
    all_val_probas = []
    all_val_ys = []
    all_xgb_importances = []
    best_score = -1
    for fold, (tr_idx, te_idx) in enumerate(pts.split(X_pool)):
        X_tr, X_te = X_pool[tr_idx], X_pool[te_idx]
        y_tr, y_te = y_pool[tr_idx], y_pool[te_idx]
        sw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        ens, score, _, _, raw_imp, vp, vy = train_fold_ensemble(
            X_tr, y_tr, X_te, y_te, sw, tune_params=tuned_params, use_tft=tft
        )
        if raw_imp is not None:
            all_xgb_importances.append(raw_imp)
        print(f"  Fold {fold + 1}: ensemble f1={score:.4f}")
        all_val_probas.append(vp)
        all_val_ys.append(vy)
        if score > best_score:
            best_score = score
            best_model = ens
    oof_probas = np.vstack(all_val_probas)
    oof_true = np.concatenate(all_val_ys)
    opt_threshold = optimize_threshold(oof_true, oof_probas)
    y_pred_full = best_model.predict(X_pool)
    print("\nPool Model Classification Report:")
    print(classification_report(y_pool, y_pred_full, digits=3))
    model_path = MODELS_DIR / f"model_pool_{class_label}.pkl"
    metadata = {
        "symbols": symbols,
        "features": FEATURE_COLS,
        "classes": [0, 1],
        "train_date": datetime.now().isoformat(),
        "n_samples": len(X_pool),
        "n_features": len(FEATURE_COLS),
        "f1_score": float(best_score),
        "model_type": "pool_ensemble",
        "optimal_threshold": opt_threshold,
        "asset_class": class_label,
    }
    tmp_path = model_path.with_suffix(".tmp")
    joblib.dump({"model": best_model, "metadata": metadata}, tmp_path)
    tmp_path.replace(model_path)
    print(f"Pool model saved to {model_path}")
    if has_calib:
        calib_path = model_path.with_suffix(".calib.npz")
        np.savez_compressed(calib_path, X=X_calib, y=y_calib)
        print(f"  Calibration holdout saved to {calib_path} (n={len(X_calib)})")
    return model_path


def train_model_for_symbol(
    symbol,
    years=3,
    tp_atr_mult=2.0,
    sl_atr_mult=1.0,
    max_hold=12,
    tune=False,
    prune=False,
    tuned_params=None,
    meta=False,
    regression=False,
    drift_stats=False,
    tft=False,
    csv_mode=False,
):
    print(f"\n{'=' * 60}")
    print(f"Training model for {symbol}")
    print(f"{'=' * 60}")

    sym_tp, sym_sl = _resolve_label_params(symbol)
    tp_atr_mult = sym_tp
    sl_atr_mult = sym_sl
    print(
        f"  {'E[R] regression' if regression else 'Binary classifier'}: "
        f"tp_atr={tp_atr_mult:.2f} sl_atr={sl_atr_mult:.2f} max_hold={max_hold}"
    )

    df = fetch_data(symbol, years, csv_mode=csv_mode)
    if df is None or len(df) < 200:
        return None

    print(f"Computing features ({len(FEATURE_COLS)} cols)...")
    df_m1 = fetch_m1_data(symbol, years, csv_mode=csv_mode)
    feature_data, full_df = prepare_features(df, symbol=symbol, m1_df=df_m1)
    print(f"Feature matrix: {feature_data.shape}")

    if regression:
        print("Computing R-multiple regression labels...")
        labels = r_multiple_labels(full_df, tp_atr_mult, sl_atr_mult, max_hold)
        aligned = pd.concat([feature_data, labels], axis=1).dropna(subset=["r_multiple"])
        X = aligned[FEATURE_COLS].values
        y_raw = aligned["r_multiple"].values
        print(f"  R-multiple range: [{y_raw.min():.3f}, {y_raw.max():.3f}] mean={y_raw.mean():.3f}")
        if np.isnan(X).any():
            print(f"  NaN in feature matrix ({np.isnan(X).sum()} cells) — filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        n_splits = min(5, max(2, len(X) // 100))
        pts = PurgedTimeSeriesSplit(n_splits=n_splits, gap=max_hold)
        best_mae = float("inf")
        best_model = None
        for fold, (tr_idx, te_idx) in enumerate(pts.split(X)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y_raw[tr_idx], y_raw[te_idx]
            ens, mae, xgb_m, lgb_m = train_regression_ensemble(X_tr, y_tr, X_te, y_te, tune_params=tuned_params)
            print(f"  Fold {fold + 1}: MAE={mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_model = ens
        final_model = best_model
        used_features = FEATURE_COLS
        model_type = "regressor"
        metadata = {
            "symbol": symbol,
            "features": used_features,
            "model_type": model_type,
            "train_date": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": len(used_features),
            "mae": float(best_mae),
            "tp_atr_mult": tp_atr_mult,
            "sl_atr_mult": sl_atr_mult,
            "max_hold": max_hold,
        }
        if drift_stats:
            stats = compute_feature_stats(aligned[FEATURE_COLS])
            metadata["feature_stats"] = stats
            print(f"  Feature drift stats computed for {len(stats)} features")
        model_path = MODELS_DIR / f"model_{symbol.replace('.', '_')}.pkl"
        tmp_path = model_path.with_suffix(".tmp")
        joblib.dump({"model": final_model, "metadata": metadata}, tmp_path)
        tmp_path.replace(model_path)
        print(f"Model saved to {model_path}")
        return model_path

    # --- Binary classifier path (unchanged) ---
    print("Computing triple-barrier labels...")
    labels = triple_barrier_labels(full_df, tp_atr_mult, sl_atr_mult, max_hold)
    aligned = pd.concat([feature_data, labels], axis=1).dropna(subset=["label"])
    aligned = aligned[aligned.index.isin(feature_data.index)]

    X = aligned[FEATURE_COLS].values
    y = aligned["label"].values

    class_dist = {c: (y == c).sum() for c in [-1, 0, 1]}
    print(f"Label distribution: {class_dist}")

    if np.isnan(X).any():
        print(f"  NaN in feature matrix ({np.isnan(X).sum()} cells) — filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    if len(np.unique(y)) < 2:
        print("  Single class only — cannot train, skipping")
        return None

    mask = y != 0
    X_dir = X[mask]
    y_dir = y[mask]
    y_dir = (y_dir == 1).astype(np.int8)

    class_dist_dir = {1: (y_dir == 1).sum(), 0: (y_dir == 0).sum()}
    print(f"Directional label distribution (no neutral): {class_dist_dir}")

    if len(np.unique(y_dir)) < 2:
        print("  Single class only — cannot train binary classifier, skipping")
        return None

    has_calib = False
    X_calib = np.array([])
    y_calib = np.array([])

    if drift_stats:
        stats = compute_feature_stats(aligned[FEATURE_COLS])

    if len(X_dir) < 200:
        print("Not enough directional samples, training on available directional data")
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X_dir, y_dir, test_size=0.2, shuffle=False)
        xgb_m, _ = train_xgb(X_train, y_train, tune_params=tuned_params)
        sw_small = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        lgb_m = train_lgb(X_train, y_train, scale_pos_weight=sw_small, tune_params=tuned_params)
        xgb_m = calibrate_model(xgb_m, X_test, y_test)
        lgb_m = calibrate_model(lgb_m, X_test, y_test)
        model = EnsembleModel(xgb_m, lgb_m)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        opt_threshold = optimize_threshold(y_test, y_proba)
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        final_model = model
        used_features = FEATURE_COLS
    else:
        print(f"Training XGBoost on {len(X_dir)} directional samples...")

        n_calib = min(200, max(50, int(len(X_dir) * 0.15)))
        has_calib = n_calib >= 50 and len(X_dir) - n_calib >= 300
        if has_calib:
            X_calib = X_dir[-n_calib:]
            y_calib = y_dir[-n_calib:]
            X_dir = X_dir[:-n_calib]
            y_dir = y_dir[:-n_calib]
            print(f"  Calibration holdout: {len(X_calib)} samples")

        n_splits = min(5, max(2, len(X_dir) // 100))
        pts = PurgedTimeSeriesSplit(n_splits=n_splits, gap=max_hold)

        best_score = -1
        best_model = None
        all_xgb_importances = []

        if tune and tuned_params is None:
            sw_global = min((y_dir == 0).sum() / max((y_dir == 1).sum(), 1), 20)
            tuned_params = tune_xgb_params(X_dir, y_dir, sw_global, n_trials=30)

        all_val_probas = []
        all_val_ys = []

        for fold, (train_idx, test_idx) in enumerate(pts.split(X_dir)):
            X_tr, X_te = X_dir[train_idx], X_dir[test_idx]
            y_tr, y_te = y_dir[train_idx], y_dir[test_idx]

            neg_count = (y_tr == 0).sum()
            pos_count = (y_tr == 1).sum()
            sw = neg_count / max(pos_count, 1)

            ensemble, score, xgb_m, lgb_m, raw_imp, val_proba, val_y = train_fold_ensemble(
                X_tr, y_tr, X_te, y_te, sw, tune_params=tuned_params, use_tft=tft
            )
            if raw_imp is not None:
                all_xgb_importances.append(raw_imp)

            print(f"  Fold {fold + 1}: ensemble f1={score:.4f}, pos_rate={y_tr.mean():.3f}")
            all_val_probas.append(val_proba)
            all_val_ys.append(val_y)

            if score > best_score:
                best_score = score
                best_model = ensemble

        oof_probas = np.vstack(all_val_probas)
        oof_true = np.concatenate(all_val_ys)
        opt_threshold = optimize_threshold(oof_true, oof_probas)

        y_pred_full = best_model.predict(X_dir)
        print("\nFinal Ensemble Classification Report:")
        print(classification_report(y_dir, y_pred_full, digits=3))

        if all_xgb_importances and prune:
            avg_imp = np.mean(all_xgb_importances, axis=0)
            feat_names = FEATURE_COLS
            kept = prune_features(feat_names, avg_imp, keep_ratio=0.7)
            print(f"  Retraining with {len(kept)} kept features...")
            kept_idx = [FEATURE_COLS.index(f) for f in kept]
            X_dir_kept = X_dir[:, kept_idx]
            from sklearn.model_selection import train_test_split

            X_k_tr, X_k_te, y_k_tr, y_k_te = train_test_split(X_dir_kept, y_dir, test_size=0.2, shuffle=False)
            neg_count = (y_k_tr == 0).sum()
            pos_count = (y_k_tr == 1).sum()
            sw = neg_count / max(pos_count, 1)
            xgb_p, _ = train_xgb(X_k_tr, y_k_tr, scale_pos_weight=sw, tune_params=tuned_params)
            lgb_p = train_lgb(X_k_tr, y_k_tr, scale_pos_weight=sw, tune_params=tuned_params)
            xgb_p = calibrate_model(xgb_p, X_k_te, y_k_te)
            lgb_p = calibrate_model(lgb_p, X_k_te, y_k_te)
            final_model = EnsembleModel(xgb_p, lgb_p)
            y_k_pred = final_model.predict(X_k_te)
            print("\nPruned Model Holdout Report:")
            print(classification_report(y_k_te, y_k_pred, digits=3))
            used_features = kept
        else:
            final_model = best_model
            used_features = FEATURE_COLS

    model_path = MODELS_DIR / f"model_{symbol.replace('.', '_')}.pkl"
    metadata = {
        "symbol": symbol,
        "features": used_features,
        "classes": [0, 1],
        "train_date": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features": len(used_features),
        "f1_score": float(best_score) if "best_score" in locals() else None,
        "model_type": "ensemble",
        "optimal_threshold": opt_threshold,
    }
    if drift_stats and "stats" in dir() and stats:
        metadata["feature_stats"] = stats
        print(f"  Feature drift stats computed for {len(stats)} features")

    tmp_path = model_path.with_suffix(".tmp")
    joblib.dump({"model": final_model, "metadata": metadata}, tmp_path)
    tmp_path.replace(model_path)
    print(f"\nModel saved to {model_path}")

    if has_calib:
        calib_path = model_path.with_suffix(".calib.npz")
        np.savez(calib_path, X=X_calib, y=y_calib)
        print(f"Calibration holdout saved to {calib_path}")

    meta_model_path = None
    if meta:
        meta_model = train_meta_labeler(
            X_dir, y_pred_full if "y_pred_full" in dir() else None, y_dir, tune_params=tuned_params
        )
        if meta_model:
            meta_model_path = model_path.with_suffix(".meta.pkl")
            joblib.dump(
                {"model": meta_model, "metadata": {"parent": symbol, "optimal_threshold": 0.5}}, meta_model_path
            )
            print(f"Meta-labeler saved to {meta_model_path}")

    xgb_m_final = final_model.xgb
    importances = xgb_m_final.feature_importances_
    feat_imp = sorted(zip(used_features, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 10 XGBoost feature importances:")
    for feat, score in feat_imp[:10]:
        print(f"  {feat}: {score:.4f}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train ML signal model for Doto MT5 bot")
    parser.add_argument(
        "--symbols", type=str, default="ETHUSD.raw,XAUUSD.raw", help="Symbols to train on (comma-separated)"
    )
    parser.add_argument("--years", type=float, default=3.0, help="Years of historical data")
    parser.add_argument("--tp-atr", type=float, default=2.0, help="Take profit ATR multiplier")
    parser.add_argument("--sl-atr", type=float, default=1.0, help="Stop loss ATR multiplier")
    parser.add_argument("--max-hold", type=int, default=12, help="Max holding period in bars")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--prune", action="store_true", help="Prune bottom 30%% features by importance")
    parser.add_argument("--retrain-all", action="store_true", help="Retrain all portfolio symbols")
    parser.add_argument("--pool", action="store_true", help="Train pool model per asset class")
    parser.add_argument("--meta", action="store_true", help="Train meta-labeler in addition to primary model")
    parser.add_argument(
        "--regression", action="store_true", help="Train E[PnL] regression model instead of binary classifier"
    )
    parser.add_argument("--drift-stats", action="store_true", help="Save per-feature stats for live drift monitoring")
    parser.add_argument(
        "--tft", action="store_true", help="Train TFT as 4th ensemble member alongside XGBoost/LightGBM"
    )
    parser.add_argument("--csv", action="store_true", help="Use pre-exported CSV data instead of MT5 terminal")
    args = parser.parse_args()

    print(f"ML Model Trainer — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(
        f"Config: tp_atr={args.tp_atr}, sl_atr={args.sl_atr}, "
        f"max_hold={args.max_hold}, years={args.years}, csv={args.csv}"
    )

    if args.retrain_all:
        symbols = [
            "XAU500.raw",
            "BTCUSD.raw",
            "NZDUSD.raw",
            "US30.raw",
            "GBPJPY.raw",
            "SOLUSD.raw",
            "XRPUSD.raw",
            "DOGUSD.raw",
        ]
        print(f"Retrain-all mode: {symbols}")
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    if not args.csv and not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    if args.pool:
        classes = set(ASSET_CLASS_MAP.get(s) for s in symbols if s in ASSET_CLASS_MAP)
        for cls in classes:
            class_syms = [s for s in symbols if ASSET_CLASS_MAP.get(s) == cls]
            if len(class_syms) >= 2:
                try:
                    train_pool_model(class_syms, args.years, args.tp_atr, args.sl_atr, args.max_hold, tft=args.tft)
                except Exception as e:
                    import traceback

                    print(f"Error training pool for {cls}: {e}")
                    traceback.print_exc()

    for symbol in symbols:
        try:
            train_model_for_symbol(
                symbol,
                args.years,
                args.tp_atr,
                args.sl_atr,
                args.max_hold,
                tune=args.tune,
                prune=args.prune,
                meta=args.meta,
                regression=args.regression,
                drift_stats=args.drift_stats,
                tft=args.tft,
                csv_mode=args.csv,
            )
        except Exception as e:
            import traceback

            print(f"Error training {symbol}: {e}")
            traceback.print_exc()

    if not args.csv:
        mt5.shutdown()
    print("\nDone.")


if __name__ == "__main__":
    main()
