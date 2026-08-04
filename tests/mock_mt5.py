"""Factory classes for fake MT5 objects — reusable across mock-based tests."""

from dataclasses import dataclass


@dataclass
class FakeAccountInfo:
    balance: float = 100000.0
    equity: float = 105000.0
    profit: float = 5000.0
    margin: float = 10000.0
    margin_free: float = 90000.0


@dataclass
class FakeTick:
    bid: float = 2000.0
    ask: float = 2000.5


@dataclass
class FakeSymbolInfo:
    trade_tick_value: float = 0.1
    trade_tick_size: float = 0.01
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    trade_stops_level: int = 50
    point: float = 0.01


@dataclass
class FakePosition:
    ticket: int = 1001
    symbol: str = "XAU500.raw"
    type: int = 0  # ORDER_TYPE_BUY
    volume: float = 0.1
    price_open: float = 2000.0
    sl: float = 1990.0
    tp: float = 2020.0
    price_current: float = 2010.0
    profit: float = 100.0
    comment: str = "TrendBot"


@dataclass
class FakeOrderSendResult:
    retcode: int = 10009  # TRADE_RETCODE_DONE
    order: int = 5001
    price: float = 2000.5
    volume: float = 0.1
    comment: str = ""


@dataclass
class FakeDeal:
    price: float = 2010.0
    profit: float = 150.0


class FakeMt5Module:
    """Stand-in for the MetaTrader5 module — exposes constants and returns fake data."""

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    TRADE_ACTION_REMOVE = 3
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16392

    def __init__(self):
        self._account_info = FakeAccountInfo()
        self._tick = FakeTick()
        self._symbol_info = FakeSymbolInfo()
        self._last_error_val = 0

    def account_info(self):
        return self._account_info

    def symbol_info_tick(self, symbol):
        return self._tick

    def symbol_info(self, symbol):
        return self._symbol_info

    def symbol_select(self, symbol, enable):
        return True

    def initialize(self, **kwargs):
        return True

    def login(self, **kwargs):
        return True

    def shutdown(self):
        pass

    def last_error(self):
        return self._last_error_val

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        import numpy as np
        import pandas as pd
        n = count
        closes = 2000 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": closes - np.random.uniform(0, 0.5, n),
            "high": closes + np.random.uniform(0.1, 1.0, n),
            "low": closes - np.random.uniform(0.1, 1.0, n),
            "close": closes,
            "tick_volume": np.random.randint(100, 10000, n),
        }).to_records(index=False)

    def history_deals_get(self, position=None):
        if position is not None:
            return (FakeDeal(),)
        return ()

    def positions_get(self, symbol=None):
        return ()

    def order_send(self, request):
        return FakeOrderSendResult()

    def terminal_info(self):
        from types import SimpleNamespace
        return SimpleNamespace(connected=True, server="DOTOGlobal-Real")
