"""Tests for train_model.py — ML model training, ensemble, threshold optimization.

Most functions require MT5 data so tests are limited to pure-logic components.
"""

import numpy as np
import pandas as pd
import pytest


class TestPurgedTimeSeriesSplit:
    def test_returns_train_test_indices(self):
        from train_model import PurgedTimeSeriesSplit
        n = 200
        X = pd.DataFrame({"feat": range(n)}, index=range(n))
        y = pd.Series(np.random.randn(n), index=range(n))
        pts = PurgedTimeSeriesSplit(n_splits=3, gap=5)
        splits = list(pts.split(X, y))
        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(test_idx) > 0
            assert max(train_idx) < min(test_idx)

    def test_gap_prevents_leakage(self):
        from train_model import PurgedTimeSeriesSplit
        n = 200
        X = pd.DataFrame({"feat": range(n)}, index=range(n))
        y = pd.Series(np.random.randn(n), index=range(n))
        pts = PurgedTimeSeriesSplit(n_splits=1, gap=5)
        train_idx, test_idx = next(pts.split(X, y))
        gap = min(test_idx) - max(train_idx)
        assert gap >= 5


class TestEnsembleModel:
    def test_predict_proba_returns_2d(self):
        import numpy as np
        from train_model import EnsembleModel
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.3, 0.7]] * len(X))
        model = EnsembleModel(MockModel(), MockModel())
        X = np.random.randn(10, 5)
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_returns_labels(self):
        from train_model import EnsembleModel
        class MockModel:
            def predict(self, X):
                return np.ones(len(X))
        model = EnsembleModel(MockModel(), MockModel())
        X = np.random.randn(5, 3)
        preds = model.predict(X)
        assert len(preds) == 5
        assert set(preds) == {1}


class TestOptimizeThreshold:
    @pytest.fixture
    def proba_2d(self):
        rng = np.random.RandomState(42)
        p = np.concatenate([rng.uniform(0.5, 1.0, 100), rng.uniform(0.0, 0.5, 100)])
        return np.column_stack([1 - p, p])

    def test_returns_threshold_in_01_range(self, proba_2d):
        from train_model import optimize_threshold
        y_true = np.array([1] * 100 + [0] * 100)
        threshold = optimize_threshold(y_true, proba_2d, rr=2.0, min_trades=1)
        assert 0.0 <= threshold <= 1.0

    def test_returns_float(self, proba_2d):
        from train_model import optimize_threshold
        y_true = np.array([1] * 100 + [0] * 100)
        result = optimize_threshold(y_true, proba_2d, rr=2.0, min_trades=1)
        assert isinstance(result, float)


class TestPruneFeatures:
    def test_keeps_top_features(self):
        from train_model import prune_features
        names = [f"feat_{i}" for i in range(10)]
        importances = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05])
        kept = prune_features(names, importances, keep_ratio=0.5)
        assert len(kept) <= 5
        assert "feat_2" in kept  # highest importance

    def test_keeps_at_least_one(self):
        from train_model import prune_features
        names = ["feat_a", "feat_b"]
        importances = np.array([0.001, 0.001])
        kept = prune_features(names, importances, keep_ratio=0.1)
        assert len(kept) >= 1
