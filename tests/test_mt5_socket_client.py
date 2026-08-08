"""Tests for the MQL5 socket bridge client."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from mt5_socket_client import (  # noqa: E402
    TIMEFRAME_D1,
    TIMEFRAME_H1,
    TIMEFRAME_H4,
    TIMEFRAME_M1,
    TIMEFRAME_M15,
    TIMEFRAME_M30,
    TIMEFRAME_MN1,
    TIMEFRAME_W1,
    MT5SocketClient,
    timeframe_seconds,
)


class TestTimeframeSeconds:
    """MT5 packs the period in the low bits and the unit in flag bits."""

    @pytest.mark.parametrize(
        "timeframe,expected",
        [
            (TIMEFRAME_M1, 60),
            (TIMEFRAME_M15, 15 * 60),
            (TIMEFRAME_M30, 30 * 60),
            (TIMEFRAME_H1, 3600),
            (TIMEFRAME_H4, 4 * 3600),
            (TIMEFRAME_D1, 24 * 3600),
            (TIMEFRAME_W1, 7 * 24 * 3600),
            (TIMEFRAME_MN1, 30 * 24 * 3600),
        ],
    )
    def test_known_timeframes(self, timeframe, expected):
        assert timeframe_seconds(timeframe) == expected

    def test_month_flag_checked_before_week_and_hour(self):
        # MN1 sets 0xC000, which also matches the 0x8000 and 0x4000 tests.
        # Ordering matters or months would decode as weeks.
        assert timeframe_seconds(TIMEFRAME_MN1) > timeframe_seconds(TIMEFRAME_W1)


class _FakeClient(MT5SocketClient):
    """Captures the wire command instead of touching a socket."""

    def __init__(self):  # noqa: D107  (deliberately skips MT5SocketClient.__init__)
        self.sent = None

    def _call_multi(self, cmd):
        self.sent = cmd
        return []


class TestCopyRatesFromDirection:
    """Regression: copy_rates_from must page BACKWARDS with a per-timeframe width.

    The original implementation hardcoded ``count * 3600`` and added it to the
    cursor, so it requested a FORWARD H1-sized window. For M1 that asked for
    `count` hours of future data and returned zero bars, which made
    fetch_rates_paged() return None for every non-H1 timeframe.
    """

    @staticmethod
    def _range_args(cmd):
        parts = cmd.split()
        assert parts[0] == "RATES_RANGE", cmd
        return int(parts[3]), int(parts[4])  # ts_from, ts_to

    def test_window_ends_at_cursor(self):
        c = _FakeClient()
        c.copy_rates_from("XAUUSD.raw", TIMEFRAME_M1, 1_000_000, 500)
        ts_from, ts_to = self._range_args(c.sent)
        assert ts_to == 1_000_000, "window must END at the cursor, not start there"
        assert ts_from < ts_to, "must page backwards"

    def test_m1_width_is_one_minute_per_bar(self):
        c = _FakeClient()
        c.copy_rates_from("XAUUSD.raw", TIMEFRAME_M1, 1_000_000, 500)
        ts_from, ts_to = self._range_args(c.sent)
        assert ts_to - ts_from == 500 * 60

    def test_h1_width_is_one_hour_per_bar(self):
        c = _FakeClient()
        c.copy_rates_from("XAUUSD.raw", TIMEFRAME_H1, 1_000_000, 500)
        ts_from, ts_to = self._range_args(c.sent)
        assert ts_to - ts_from == 500 * 3600

    def test_m15_width_differs_from_h1(self):
        m15, h1 = _FakeClient(), _FakeClient()
        m15.copy_rates_from("X", TIMEFRAME_M15, 1_000_000, 100)
        h1.copy_rates_from("X", TIMEFRAME_H1, 1_000_000, 100)
        assert self._range_args(m15.sent) != self._range_args(h1.sent)

    def test_accepts_datetime(self):
        from datetime import datetime

        c = _FakeClient()
        dt = datetime(2026, 1, 1, 12, 0, 0)
        c.copy_rates_from("X", TIMEFRAME_M1, dt, 10)
        ts_from, ts_to = self._range_args(c.sent)
        assert ts_to == int(dt.timestamp())
        assert ts_to - ts_from == 10 * 60
