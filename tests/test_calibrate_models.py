"""Tests for calibrate_models.py — isotonic calibration of ML ensemble probabilities."""

import joblib
import numpy as np


class MockModel:
    """Module-level so joblib can pickle it."""
    xgb = None
    lgb = None


class TestCalibrateModelFile:
    def test_missing_calib_file_returns_false(self, tmp_path):
        from calibrate_models import calibrate_model_file
        model_path = tmp_path / "model_dummy.pkl"
        model_path.write_text("dummy")
        assert calibrate_model_file(model_path) is False

    def test_not_a_model_file_returns_false(self, tmp_path):
        from calibrate_models import calibrate_model_file
        model_path = tmp_path / "model_dummy.pkl"
        calib_path = model_path.with_suffix(".calib.npz")
        np.savez(calib_path, X=np.array([[1.0]]), y=np.array([1]))
        model_path.write_text("not a pickle")
        assert calibrate_model_file(model_path) is False

    def test_calibration_set_too_small_returns_false(self, tmp_path):
        from calibrate_models import calibrate_model_file
        model_path = tmp_path / "model_dummy.pkl"
        calib_path = model_path.with_suffix(".calib.npz")
        np.savez(calib_path, X=np.ones((5, 3)), y=np.ones(5))
        joblib.dump({"model": MockModel(), "metadata": {}}, model_path)
        assert calibrate_model_file(model_path) is False
