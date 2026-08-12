"""Tests for drift_detector.py — warm-start queue."""


class TestWarmstartQueue:
    def test_schedule_and_consume_returns_pending(self):
        from drift_detector import consume_warmstart_queue, schedule_warmstart
        schedule_warmstart("SYM1")
        schedule_warmstart("SYM2")
        pending = consume_warmstart_queue()
        assert "SYM1" in pending
        assert "SYM2" in pending

    def test_consume_clears_queue(self):
        from drift_detector import consume_warmstart_queue, schedule_warmstart
        schedule_warmstart("SYM")
        consume_warmstart_queue()
        assert consume_warmstart_queue() == []
