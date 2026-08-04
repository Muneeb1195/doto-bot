"""Tests for tft_model.py — TFTClassifier training/inference on synthetic data."""

import sys

sys.path.insert(0, "bot")

import numpy as np
import pytest

pytest.importorskip("torch")

from tft_model import TFTClassifier, _build_sequences, train_tft


def _separable_data(n=160, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    # Label depends on a smoothed running signal so sequences are informative.
    signal = X[:, 0] + 0.5 * X[:, 1]
    y = (signal > 0).astype(int)
    return X, y


class TestBuildSequences:
    def test_shapes(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        seqs, idxs = _build_sequences(X, seq_len=5)
        assert seqs.shape[1] == 5
        assert seqs.shape[2] == 2
        assert len(seqs) == len(idxs)

    def test_short_input(self):
        X = np.arange(6, dtype=float).reshape(3, 2)
        seqs, idxs = _build_sequences(X, seq_len=5)
        assert seqs.shape[0] == 1
        assert list(idxs) == [0]


class TestTFTClassifier:
    def test_fit_sets_fitted(self):
        X, y = _separable_data()
        clf = TFTClassifier(n_features=X.shape[1], seq_len=10)
        clf.fit(X, y)
        assert clf._fitted is True

    def test_predict_proba_shape_and_bounds(self):
        X, y = _separable_data()
        clf = TFTClassifier(n_features=X.shape[1], seq_len=10)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_returns_binary(self):
        X, y = _separable_data()
        clf = TFTClassifier(n_features=X.shape[1], seq_len=10)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        clf = TFTClassifier(n_features=4, seq_len=10)
        with pytest.raises(RuntimeError):
            clf.predict_proba(np.zeros((30, 4), dtype=np.float32))

    def test_classes(self):
        clf = TFTClassifier(n_features=4)
        assert list(clf.classes_) == [0, 1]

    def test_get_params_roundtrip(self):
        clf = TFTClassifier(n_features=7, seq_len=15, hidden_size=32)
        params = clf.get_params()
        assert params["n_features"] == 7
        assert params["seq_len"] == 15
        assert params["hidden_size"] == 32

    def test_train_tft_helper(self):
        X, y = _separable_data()
        clf = train_tft(X, y)
        assert clf._fitted is True
        assert clf.predict_proba(X).shape == (len(X), 2)
