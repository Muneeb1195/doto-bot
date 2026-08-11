import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
import state as _st
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from ml_features import FEATURE_COLS, compute_feature_stats, prepare_features, triple_barrier_labels
from train_model import EnsembleModel, EnsembleRegressor, _CalibratedWrapper

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
_RETRAIN_BARS = 2000
_MIN_RETRAIN_SAMPLES = 100


def _resolve_label_params(cfg, symbol):
    sym_section = f"STRATEGY:{symbol}"
    if cfg.has_section(sym_section):
        sl_mult = cfg.getfloat(sym_section, "atr_sl_multiplier", fallback=None)
        rr = cfg.getfloat(sym_section, "risk_reward_ratio", fallback=None)
        if sl_mult is not None and rr is not None:
            return sl_mult * rr, sl_mult
    sl_mult = cfg.getfloat("STRATEGY", "atr_sl_multiplier", fallback=1.0)
    rr = cfg.getfloat("STRATEGY", "risk_reward_ratio", fallback=2.0)
    return sl_mult * rr, sl_mult


def warmstart_model(symbol, cfg=None, max_hold=12):
    safe_symbol = symbol.replace(".", "_")
    model_path = MODELS_DIR / f"model_{safe_symbol}.pkl"
    if not model_path.exists():
        logging.warning(f"[{symbol}] No existing model to warm-start at {model_path}")
        return False

    try:
        model_data = joblib.load(model_path)
    except Exception as e:
        logging.error(f"[{symbol}] Failed to load model for warm-start: {e}")
        return False

    model = model_data["model"]
    metadata = model_data.get("metadata", {})

    if not isinstance(model, EnsembleModel):
        logging.warning(f"[{symbol}] Warm-start requires EnsembleModel, got {type(model).__name__}")
        return False

    import configparser

    if cfg is None:
        cfg = configparser.ConfigParser()
        cfg.read(BASE_DIR / "config" / "settings.ini")

    from mt5_connect import mt5_call

    end = pd.Timestamp.now()
    start = end - pd.Timedelta(hours=_RETRAIN_BARS)
    rates = mt5_call(mt5.copy_rates_range, symbol, mt5.TIMEFRAME_H1, start.to_pydatetime(), end.to_pydatetime(), _timeout=60)
    if rates is None or len(rates) < _RETRAIN_BARS // 2:
        logging.warning(f"[{symbol}] Insufficient data for warm-start ({len(rates) if rates is not None else 0})")
        return False

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    m1_df = None
    try:
        m1_end = end
        m1_start = end - pd.Timedelta(hours=min(_RETRAIN_BARS, 720))
        m1_rates = mt5_call(mt5.copy_rates_range, symbol, mt5.TIMEFRAME_M1, m1_start.to_pydatetime(), m1_end.to_pydatetime(), _timeout=120)
        if m1_rates is not None and len(m1_rates) > 0:
            m1_df = pd.DataFrame(m1_rates)
            m1_df["time"] = pd.to_datetime(m1_df["time"], unit="s")
    except Exception:
        logging.debug(f"[{symbol}] M1 fetch for warm-start failed — of_* will be 0.0")

    feature_data, full_df = prepare_features(df, symbol=symbol, m1_df=m1_df)
    tp_atr, sl_atr = _resolve_label_params(cfg, symbol)
    labels = triple_barrier_labels(full_df, tp_atr, sl_atr, max_hold)

    aligned = pd.concat([feature_data, labels], axis=1).dropna(subset=["label"])
    aligned = aligned[aligned.index.isin(feature_data.index)]
    X = aligned[FEATURE_COLS].values
    y_directional = aligned["label"].values
    mask = y_directional != 0
    X_dir = X[mask]
    y_dir = (y_directional[mask] == 1).astype(np.int8)

    if len(X_dir) < _MIN_RETRAIN_SAMPLES:
        logging.info(
            f"[{symbol}] Warm-start: only {len(X_dir)} directional samples (need {_MIN_RETRAIN_SAMPLES}) — skipping"
        )  # noqa: E501
        return False

    xgb_model = model.xgb
    lgb_model = model.lgb

    try:
        from sklearn.calibration import CalibratedClassifierCV

        def _unwrap(m):
            # The default train_model.py path wraps each member in a
            # _CalibratedWrapper (has .base_model). calibrate_models.py uses
            # sklearn CalibratedClassifierCV instead. Unwrap either so we can
            # read the underlying booster's params for warm-starting; without
            # the .base_model branch, get_params() below raises AttributeError
            # and the whole drift retrain silently fails (agent audit D1).
            if hasattr(m, "base_model"):
                return m.base_model
            if hasattr(m, "base_estimator"):
                return m.base_estimator
            if getattr(m, "calibrated_classifiers_", None):
                return m.calibrated_classifiers_[0].base_estimator
            return m

        raw_xgb = _unwrap(xgb_model)
        raw_lgb = _unwrap(lgb_model)

        xgb_params = raw_xgb.get_params()
        xgb_params.pop("n_jobs", None)
        xgb_new = xgb.XGBClassifier(**xgb_params)
        xgb_new.fit(X_dir, y_dir, xgb_model=raw_xgb)

        lgb_params = raw_lgb.get_params()
        lgb_params.pop("n_jobs", None)
        lgb_new = lgb.LGBMClassifier(**lgb_params)
        lgb_new.fit(X_dir, y_dir, init_model=raw_lgb)

        if len(X_dir) >= 100:
            from sklearn.calibration import CalibratedClassifierCV

            n_cal = min(100, len(X_dir) // 5)
            xgb_new = CalibratedClassifierCV(xgb_new, cv="prefit", method="isotonic")
            xgb_new.fit(X_dir[:n_cal], y_dir[:n_cal])
            lgb_new = CalibratedClassifierCV(lgb_new, cv="prefit", method="isotonic")
            lgb_new.fit(X_dir[:n_cal], y_dir[:n_cal])

        # Preserve the TFT member (if the original ensemble had one) so a
        # warm-started model keeps the same 3-member structure as the
        # model it replaces (all 7 portfolio models were trained --tft).
        warm_model = EnsembleModel(xgb_new, lgb_new, tft_model=getattr(model, "tft", None))

        y_pred = warm_model.predict(X_dir)
        acc = (y_pred == y_dir).mean()
        metadata["feature_stats"] = compute_feature_stats(feature_data[FEATURE_COLS])
        metadata["last_warmstart"] = pd.Timestamp.now().isoformat()
        metadata["warmstart_samples"] = len(X_dir)
        metadata["warmstart_acc"] = float(acc)

        tmp_path = model_path.with_suffix(".tmp")
        joblib.dump({"model": warm_model, "metadata": metadata}, tmp_path)
        tmp_path.replace(model_path)
        logging.info(f"[{symbol}] Warm-start complete: {len(X_dir)} samples, acc={acc:.3f}")

        # Reload the freshly written model into the running bot's in-memory
        # cache so drift-driven warm-start actually takes effect WITHOUT a restart.
        # Otherwise the new file on disk is ignored until the bot is rebooted
        # (main.py only loads models at startup via load_ml_models).
        try:
            import __main__

            __main__.EnsembleModel = EnsembleModel
            __main__.EnsembleRegressor = EnsembleRegressor
            __main__._CalibratedWrapper = _CalibratedWrapper
            reloaded = joblib.load(model_path)
            _st._ml_models[symbol] = reloaded
            logging.info(f"[{symbol}] Warm-start model reloaded into memory")
        except Exception as e:
            logging.warning(f"[{symbol}] Warm-start model saved but in-memory reload failed: {e}")
        return True

    except Exception as e:
        logging.error(f"[{symbol}] Warm-start failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_warmstart_pipeline(symbols, cfg=None):
    logging.info(f"Warm-start pipeline: {symbols}")
    results = {}
    for symbol in symbols:
        ok = warmstart_model(symbol, cfg=cfg)
        results[symbol] = "ok" if ok else "failed"
    return results


if __name__ == "__main__":
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read(BASE_DIR / "config" / "settings.ini")
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else []
    if not symbols:
        print("Usage: python drift_retrain.py SYMBOL1,SYMBOL2,...")
        sys.exit(1)
    results = run_warmstart_pipeline(symbols, cfg)
    mt5.shutdown()
    for sym, status in results.items():
        print(f"  {sym}: {status}")
