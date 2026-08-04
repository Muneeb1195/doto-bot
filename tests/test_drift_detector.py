"""Tests for drift_detector.py — PSI, confidence drift, warm-start queue."""

import numpy as np


class TestComputePsi:
    def test_identical_distributions_psi_zero(self):
        from drift_detector import compute_psi
        data = np.random.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_different_distributions_psi_positive(self):
        from drift_detector import compute_psi
        expected = np.random.randn(1000)
        actual = np.random.randn(1000) + 5
        psi = compute_psi(expected, actual)
        assert psi > 0.1

    def test_short_input_returns_zero(self):
        from drift_detector import compute_psi
        assert compute_psi([1, 2], [3, 4]) == 0.0

    def test_nan_values_handled(self):
        from drift_detector import compute_psi
        expected = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        actual = np.array([1.5, 2.5, 3.5, np.nan, 5.5])
        psi = compute_psi(expected, actual)
        assert isinstance(psi, float)
        assert 0.0 <= psi < 10

    def test_constant_input_returns_zero(self):
        from drift_detector import compute_psi
        expected = np.ones(100) * 42
        actual = np.ones(100) * 42
        psi = compute_psi(expected, actual)
        assert psi == 0.0

    def test_psi_symmetric_not_guaranteed(self):
        from drift_detector import compute_psi
        a = np.random.randn(1000)
        b = a + 1.0
        psi_ab = compute_psi(a, b)
        psi_ba = compute_psi(b, a)
        assert abs(psi_ab - psi_ba) < 2.0


class TestCheckConfidenceDrift:
    def test_no_drift_when_confidence_stable(self):
        from drift_detector import check_confidence_drift
        assert check_confidence_drift("TEST", 0.50, 0.48) is False

    def test_drift_when_confidence_drops(self):
        from drift_detector import check_confidence_drift
        assert check_confidence_drift("TEST", 0.50, 0.30) is True

    def test_zero_baseline_returns_false(self):
        from drift_detector import check_confidence_drift
        assert check_confidence_drift("TEST", 0.0, 0.30) is False

    def test_negative_baseline_returns_false(self):
        from drift_detector import check_confidence_drift
        assert check_confidence_drift("TEST", -1.0, 0.30) is False


class TestWarmstartQueue:
    def test_schedule_and_pending(self):
        from drift_detector import has_pending_warmstart, schedule_warmstart
        schedule_warmstart("TEST")
        assert has_pending_warmstart("TEST") is True

    def test_consume_returns_pending(self):
        from drift_detector import consume_warmstart_queue, schedule_warmstart
        schedule_warmstart("SYM1")
        schedule_warmstart("SYM2")
        pending = consume_warmstart_queue()
        assert "SYM1" in pending
        assert "SYM2" in pending

    def test_consume_clears_queue(self):
        from drift_detector import consume_warmstart_queue, has_pending_warmstart, schedule_warmstart
        schedule_warmstart("SYM")
        consume_warmstart_queue()
        assert has_pending_warmstart("SYM") is False
