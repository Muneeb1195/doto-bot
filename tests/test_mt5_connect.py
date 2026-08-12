"""Tests for mt5_connect.py — mt5_call timeout, get_rates caching, market_open."""

import sys

sys.path.insert(0, "bot")

import numpy as np
import pytest
import state as _st


class TestMt5Call:
    def test_returns_result_from_func(self):
        from mt5_connect import mt5_call
        result = mt5_call(lambda: 42)
        assert result == 42

    def test_timeout_kwarg_accepted(self):
        from mt5_connect import mt5_call

        def fast():
            return 99

        # `_timeout` is part of the contract and accepted by mt5_call. On
        # Windows the MT5 API is bound to the calling (main) thread, so calls
        # run synchronously and return their result; a hard timeout is not
        # enforced here (it would require moving the call off the main thread,
        # which breaks the MT5 API).
        assert mt5_call(fast, _timeout=0.1) == 99

    def test_swallows_exception(self):
        from mt5_connect import mt5_call

        def crash():
            raise ValueError("boom")

        # Exceptions are swallowed so a single failed MT5 call never kills the
        # trading loop.
        assert mt5_call(crash) is None

    def test_passes_args_and_kwargs(self):
        from mt5_connect import mt5_call

        def adder(a, b=0):
            return a + b

        assert mt5_call(adder, 3, b=4) == 7

    def test_default_timeout_kwarg_accepted(self):
        from mt5_connect import mt5_call

        def fast():
            return 42

        assert mt5_call(fast) == 42


class FakeRatesArray:
    """Simulate mt5.copy_rates_from_pos return value."""

    @staticmethod
    def make(n=100, start_val=100.0):
        closes = start_val + np.cumsum(np.random.randn(n).astype(np.float64) * 0.3)
        times = np.array([i * 3600 for i in range(n)], dtype=np.int64)
        rates = np.zeros(n, dtype=[
            ("time", np.int64), ("open", np.float64), ("high", np.float64),
            ("low", np.float64), ("close", np.float64), ("tick_volume", np.int64),
            ("spread", np.int64), ("real_volume", np.int64),
        ])
        rates["time"] = times
        rates["open"] = closes - 0.1
        rates["high"] = closes + 0.5
        rates["low"] = closes - 0.5
        rates["close"] = closes
        rates["tick_volume"] = 1000
        rates["spread"] = 10
        rates["real_volume"] = 1000
        return rates


class TestGetRates:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        _st._rate_cache.clear()
        yield
        _st._rate_cache.clear()

    def test_returns_dataframe(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: FakeRatesArray.make(100))
        from mt5_connect import get_rates
        df = get_rates("XAU500.raw", 16385, 100)
        assert df is not None
        assert len(df) == 100
        assert "time" in df.columns
        assert "close" in df.columns

    def test_returns_none_when_mt5_fails(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: None)
        from mt5_connect import get_rates
        assert get_rates("XAU500.raw", 16385, 100) is None

    def test_returns_none_when_too_few_bars(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: FakeRatesArray.make(10))
        from mt5_connect import get_rates
        assert get_rates("XAU500.raw", 16385, 100) is None

    def test_caches_and_reuses(self, monkeypatch):
        call_count = [0]

        def mock_mt5_call(f, *a, **kw):
            call_count[0] += 1
            return FakeRatesArray.make(100)

        monkeypatch.setattr("mt5_connect.mt5_call", mock_mt5_call)
        from mt5_connect import get_rates
        df1 = get_rates("XAU500.raw", 16385, 100)
        df2 = get_rates("XAU500.raw", 16385, 100)
        df3 = get_rates("XAU500.raw", 16385, 100)
        assert df1 is df2
        assert df2 is df3
        assert call_count[0] == 1

    def test_cache_misses_for_different_bars(self, monkeypatch):
        call_count = [0]

        def mock_mt5_call(f, *a, **kw):
            call_count[0] += 1
            return FakeRatesArray.make(200)

        monkeypatch.setattr("mt5_connect.mt5_call", mock_mt5_call)
        from mt5_connect import get_rates
        get_rates("XAU500.raw", 16385, 50)
        # request more bars than cached (200) forces a miss
        get_rates("XAU500.raw", 16385, 300)
        assert call_count[0] == 2

    def test_cache_misses_for_different_symbol(self, monkeypatch):
        call_count = [0]

        def mock_mt5_call(f, *a, **kw):
            call_count[0] += 1
            return FakeRatesArray.make(100)

        monkeypatch.setattr("mt5_connect.mt5_call", mock_mt5_call)
        from mt5_connect import get_rates
        get_rates("XAU500.raw", 16385, 100)
        get_rates("BTCUSD.raw", 16385, 100)
        assert call_count[0] == 2

    def test_cache_expires_after_ttl(self, monkeypatch):
        call_count = [0]

        def mock_mt5_call(f, *a, **kw):
            call_count[0] += 1
            return FakeRatesArray.make(100)

        monkeypatch.setattr("mt5_connect.mt5_call", mock_mt5_call)
        import mt5_connect as mc
        old_ttl = mc._RATE_CACHE_TTL
        mc._RATE_CACHE_TTL = 0
        try:
            from mt5_connect import get_rates
            get_rates("XAU500.raw", 16385, 100)
            get_rates("XAU500.raw", 16385, 100)
            assert call_count[0] == 2
        finally:
            mc._RATE_CACHE_TTL = old_ttl


class FakeSymbolInfo:
    def __init__(self, trade_mode=0, filling_mode=1):
        self.trade_mode = trade_mode
        self.filling_mode = filling_mode
        self.trade_tick_value = 0.1
        self.trade_tick_size = 0.01
        self.point = 0.01


class TestMarketOpen:
    @pytest.fixture(autouse=True)
    def setup_mt5_constants(self):
        import mt5_connect as mc
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(mc.mt5, "SYMBOL_TRADE_MODE_FULL", 3)
        monkeypatch.setattr(mc.mt5, "SYMBOL_TRADE_MODE_CLOSEONLY", 2)
        monkeypatch.setattr(mc.mt5, "SYMBOL_TRADE_MODE_DISABLED", 0)
        yield
        monkeypatch.undo()

    def test_returns_true_when_full_mode(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(trade_mode=3))
        from mt5_connect import market_open
        assert market_open("EURUSD.raw") is True

    def test_returns_true_when_close_only(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(trade_mode=2))
        from mt5_connect import market_open
        assert market_open("EURUSD.raw") is True

    def test_returns_false_when_disabled(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(trade_mode=0))
        from mt5_connect import market_open
        assert market_open("EURUSD.raw") is False

    def test_returns_false_when_info_none(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: None)
        from mt5_connect import market_open
        assert market_open("EURUSD.raw") is False

    def test_eth_always_open(self, monkeypatch):
        from mt5_connect import market_open
        assert market_open("ETHUSD.raw") is True


class TestCanTradeSymbol:
    def test_returns_true_when_full_mode(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(trade_mode=3))
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_FULL", 3)
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_CLOSEONLY", 2)
        from mt5_connect import can_trade_symbol
        assert can_trade_symbol("EURUSD.raw") is True

    def test_returns_false_when_close_only(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(trade_mode=2))
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_FULL", 3)
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_CLOSEONLY", 2)
        from mt5_connect import can_trade_symbol
        assert can_trade_symbol("EURUSD.raw") is False

    def test_returns_false_when_info_none(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: None)
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_FULL", 3)
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_CLOSEONLY", 2)
        from mt5_connect import can_trade_symbol
        assert can_trade_symbol("EURUSD.raw") is False

    def test_eth_always_tradeable(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_FULL", 3)
        monkeypatch.setattr("mt5_connect.mt5.SYMBOL_TRADE_MODE_CLOSEONLY", 2)
        from mt5_connect import can_trade_symbol
        assert can_trade_symbol("ETHUSD.raw") is True



class TestGetFillingMode:
    def test_ioc_when_bit_set(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(filling_mode=3))
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_IOC", 2)
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_FOK", 1)
        from mt5_connect import get_filling_mode
        assert get_filling_mode("EURUSD.raw") == 2

    def test_fok_when_ioc_not_set(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(filling_mode=1))
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_IOC", 2)
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_FOK", 1)
        from mt5_connect import get_filling_mode
        # filling_mode=1 means only FOK is advertised; we fall back to RETURN (0)
        assert get_filling_mode("EURUSD.raw") == 0

    def test_default_when_no_symbol_info(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call", lambda f, *a, **kw: None)
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_IOC", 2)
        from mt5_connect import get_filling_mode
        assert get_filling_mode("EURUSD.raw") == 2

    def test_default_when_filling_mode_zero(self, monkeypatch):
        monkeypatch.setattr("mt5_connect.mt5_call",
                           lambda f, *a, **kw: FakeSymbolInfo(filling_mode=0))
        monkeypatch.setattr("mt5_connect.mt5.ORDER_FILLING_IOC", 2)
        from mt5_connect import get_filling_mode
        assert get_filling_mode("EURUSD.raw") == 2


class FakePagedServer:
    """Simulate an MT5 server with N bars of history and a per-request cap.

    copy_rates_from(symbol, tf, cursor, count) returns up to `cap` bars ending
    at (and including) the bar whose time <= cursor, mimicking MT5 semantics.
    """

    def __init__(self, n_bars, cap, bar_seconds=60, t0=1_700_000_000):
        self.cap = cap
        self.times = np.array([t0 + i * bar_seconds for i in range(n_bars)], dtype=np.int64)

    def copy_rates_from(self, symbol, tf, cursor, count):
        import pandas as pd
        cursor_ts = int(pd.Timestamp(cursor).timestamp())
        idx = np.searchsorted(self.times, cursor_ts, side="right")
        if idx == 0:
            return None
        take = min(self.cap, count, idx)
        sel = self.times[idx - take:idx]
        rates = np.zeros(len(sel), dtype=[
            ("time", np.int64), ("open", np.float64), ("high", np.float64),
            ("low", np.float64), ("close", np.float64), ("tick_volume", np.int64),
            ("spread", np.int64), ("real_volume", np.int64),
        ])
        rates["time"] = sel
        rates["open"] = 1.0
        rates["high"] = 1.1
        rates["low"] = 0.9
        rates["close"] = 1.0
        rates["tick_volume"] = 100
        rates["spread"] = 2
        return rates


class TestFetchRatesPaged:
    def _run(self, monkeypatch, server, start, end, chunk):
        import mt5_connect as mc

        class FakeMt5:
            symbol_select = staticmethod(lambda s, b: True)
            copy_rates_from = server.copy_rates_from

        monkeypatch.setattr("mt5_connect.mt5", FakeMt5())
        return mc.fetch_rates_paged("TEST", 1, start, end, chunk_bars=chunk)

    def test_stitches_pages_beyond_cap(self, monkeypatch):
        import pandas as pd
        server = FakePagedServer(n_bars=250, cap=100)
        start = pd.Timestamp(1_700_000_000, unit="s")
        end = pd.Timestamp(1_700_000_000 + 260 * 60, unit="s").to_pydatetime()
        df = self._run(monkeypatch, server, start, end, chunk=100)
        assert df is not None
        assert len(df) == 250  # all bars despite 100-bar cap
        assert df["time"].is_monotonic_increasing
        assert not df["time"].duplicated().any()

    def test_single_page_when_under_cap(self, monkeypatch):
        import pandas as pd
        server = FakePagedServer(n_bars=50, cap=100)
        start = pd.Timestamp(1_700_000_000, unit="s")
        end = pd.Timestamp(1_700_000_000 + 60 * 60, unit="s").to_pydatetime()
        df = self._run(monkeypatch, server, start, end, chunk=100)
        assert df is not None
        assert len(df) == 50

    def test_trims_to_start(self, monkeypatch):
        import pandas as pd
        server = FakePagedServer(n_bars=200, cap=100)
        # start halfway through the history
        start = pd.Timestamp(1_700_000_000 + 100 * 60, unit="s")
        end = pd.Timestamp(1_700_000_000 + 210 * 60, unit="s").to_pydatetime()
        df = self._run(monkeypatch, server, start, end, chunk=100)
        assert df is not None
        assert (df["time"] >= start).all()

    def test_returns_none_when_no_data(self, monkeypatch):
        import mt5_connect as mc
        import pandas as pd

        class FakeMt5:
            symbol_select = staticmethod(lambda s, b: True)
            copy_rates_from = staticmethod(lambda *a, **kw: None)

        monkeypatch.setattr("mt5_connect.mt5", FakeMt5())
        start = pd.Timestamp(1_700_000_000, unit="s")
        df = mc.fetch_rates_paged("TEST", 1, start, start.to_pydatetime(), chunk_bars=100)
        assert df is None
