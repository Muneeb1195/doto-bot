#!/usr/bin/env python3
"""Offline probability calibration for trained Doto ML ensemble models.

Usage (Linux/Python, outside Wine):
    python calibrate_models.py --symbols ETHUSD.raw,XAU500.raw
    python calibrate_models.py --all
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Must be importable after sys.path for joblib to resolve EnsembleModel references
sys.path.insert(0, str(Path(__file__).parent.resolve()))
if __name__ == "__main__":
    from train_model import EnsembleModel, _CalibratedWrapper

    import __main__

    __main__.EnsembleModel = EnsembleModel  # type: ignore[attr-defined]
    __main__._CalibratedWrapper = _CalibratedWrapper  # type: ignore[attr-defined]


def _unwrap(m):
    """Unwrap _CalibratedWrapper to get the base sklearn-compatible model."""
    while hasattr(m, "base_model"):
        m = m.base_model
    return m


def calibrate_model_file(model_path):
    calib_path = model_path.with_suffix(".calib.npz")
    if not calib_path.exists():
        print(f"  SKIP {model_path.name}: no calibration holdout found ({calib_path.name} missing)")
        return False
    try:
        data = joblib.load(model_path)
    except Exception as e:
        print(f"  FAIL {model_path.name}: cannot load model ({e})")
        return False
    model = data["model"]
    if not hasattr(model, "xgb") or not hasattr(model, "lgb"):
        print(f"  SKIP {model_path.name}: not an EnsembleModel")
        return False
    calib = np.load(calib_path)
    X_calib = calib["X"]
    y_calib = calib["y"]
    if len(X_calib) < 10:
        print(f"  SKIP {model_path.name}: calibration set too small ({len(X_calib)} samples)")
        return False
    # Unwrap pre-calibration so CalibratedClassifierCV gets a fit-capable estimator
    raw_xgb = _unwrap(model.xgb)
    raw_lgb = _unwrap(model.lgb)
    try:
        xgb_cal = CalibratedClassifierCV(estimator=raw_xgb, method="isotonic")
        xgb_cal.fit(X_calib, y_calib)
        lgb_cal = CalibratedClassifierCV(estimator=raw_lgb, method="isotonic")
        lgb_cal.fit(X_calib, y_calib)
    except Exception as e:
        print(f"  FAIL {model_path.name}: calibration error ({e})")
        return False
    from train_model import EnsembleModel as _EM

    data["model"] = _EM(xgb_cal, lgb_cal)
    data["metadata"]["calibrated"] = True
    data["metadata"]["calibration_date"] = __import__("datetime").datetime.now().isoformat()
    data["metadata"]["calibration_method"] = "isotonic"
    data["metadata"]["calibration_samples"] = len(X_calib)
    tmp_path = model_path.with_suffix(".tmp")
    joblib.dump(data, tmp_path)
    tmp_path.replace(model_path)
    model.predict_proba(X_calib[:5])
    raw_proba = np.array([raw_xgb.predict_proba(X_calib[:5]), raw_lgb.predict_proba(X_calib[:5])]).mean(axis=0)
    calib_proba = data["model"].predict_proba(X_calib[:5])
    delta = np.abs(calib_proba - raw_proba).mean()
    print(f"  OK {model_path.name}: {len(X_calib)} calib samples, mean proba shift={delta:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Calibrate ML ensemble probabilities")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to calibrate")
    parser.add_argument("--all", action="store_true", help="Calibrate all models with calibration holdout")
    args = parser.parse_args()
    if args.all:
        patterns = sorted(MODELS_DIR.glob("model_*.pkl"))
    elif args.symbols:
        patterns = []
        for sym in args.symbols.split(","):
            p = MODELS_DIR / f"model_{sym.replace('.', '_')}.pkl"
            if p.exists():
                patterns.append(p)
    else:
        print("Specify --symbols or --all")
        return
    if not patterns:
        print("No model files found")
        return
    ok = 0
    fail = 0
    for mp in patterns:
        if calibrate_model_file(mp):
            ok += 1
        else:
            fail += 1
    print(f"\nCalibration complete: {ok} OK, {fail} failed/total {len(patterns)}")


if __name__ == "__main__":
    main()
