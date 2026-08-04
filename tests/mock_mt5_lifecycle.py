"""Stateful MT5 simulator — tracks positions, orders, deals across calls.

Used by integration tests to simulate full position lifecycles:
open → modify SL/TP → partial close → full close.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 2
TRADE_ACTION_PENDING = 5
ORDER_TIME_GTC = 0
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_FAIL = 10004
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16392
SYMBOL_TRADE_MODE_DISABLED = 0
SYMBOL_TRADE_MODE_CLOSEONLY = 1
SYMBOL_TRADE_MODE_FULL = 2
ORDER_FILLING_RETURN = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_FOK = 2


@dataclass
class SimPosition:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float = 0.0
    profit: float = 0.0
    comment: str = "TrendBot"
    time: int = 0
    magic: int = 20240706


@dataclass
class SimOrder:
    ticket: int
    symbol: str
    type: int
    volume: float
    price: float
    sl: float
    tp: float
    comment: str = "TrendBot"
    time_done: int = 0
    time_setup: int = 0


@dataclass
class SimDeal:
    ticket: int
    order: int
    position_id: int
    symbol: str
    type: int
    price: float
    volume: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0
    magic: int = 20240706
    time: int = 0
    comment: str = ""


@dataclass
class SimTick:
    bid: float = 2000.0
    ask: float = 2000.5


@dataclass
class SimSymbolInfo:
    trade_tick_value: float = 0.1
    trade_tick_size: float = 0.01
    volume_step: float = 0.01
    volume_min: float = 0.01
    volume_max: float = 100.0
    trade_stops_level: int = 50
    point: float = 0.01
    trade_mode: int = SYMBOL_TRADE_MODE_FULL
    filling_mode: int = ORDER_FILLING_IOC


@dataclass
class SimAccountInfo:
    balance: float = 100000.0
    equity: float = 105000.0
    profit: float = 5000.0
    margin: float = 10000.0
    margin_free: float = 90000.0


class Mt5Simulator:
    """Stateful MT5 mock. Tracks positions, orders, deals.

    Use `install()` to replace MetaTrader5 in sys.modules, or call
    methods directly and route via patch('bot.mt5_connect.mt5_call', ...).
    """

    def __init__(self):
        self.positions: dict[int, SimPosition] = {}
        self.orders: dict[int, SimOrder] = {}
        self.deals: list[SimDeal] = []
        self._next_ticket_val = 1000
        self._tick = SimTick()
        self._symbol_info = SimSymbolInfo()
        self._account_info = SimAccountInfo()
        self._last_error_val = 0
        self._rates_cache: dict = {}
        self._now = int(datetime.now().timestamp())

    # --- MT5 api surface ---

    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 2
    TRADE_ACTION_PENDING = 5
    TRADE_ACTION_REMOVE = 3
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_FAIL = 10004
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 16385
    TIMEFRAME_H4 = 16388
    TIMEFRAME_D1 = 16392
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_CLOSEONLY = 1
    SYMBOL_TRADE_MODE_FULL = 2
    ORDER_FILLING_RETURN = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 2

    def account_info(self):
        return self._account_info

    def symbol_info_tick(self, symbol):
        return self._tick

    def symbol_info(self, symbol):
        si = self._symbol_info
        si.filling_mode = ORDER_FILLING_IOC
        return si

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

    def terminal_info(self):
        from types import SimpleNamespace
        return SimpleNamespace(connected=True, server="DOTOGlobal-Real")

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        key = (symbol, timeframe)
        if key in self._rates_cache:
            cached = self._rates_cache[key]
            if len(cached) >= count:
                return cached.iloc[:count].to_records(index=False)
        n = max(count, 100)
        rng = np.random.default_rng(42)
        closes = 2000 + np.cumsum(rng.normal(0, 0.5, n))
        now = pd.Timestamp.now()
        df = pd.DataFrame({
            "time": pd.date_range(now - pd.Timedelta(hours=n - 1), periods=n, freq="h"),
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0.1, 1.0, n),
            "low": closes - rng.uniform(0.1, 1.0, n),
            "close": closes,
            "tick_volume": rng.integers(100, 10000, n),
            "spread": np.full(n, 5),
        })
        self._rates_cache[key] = df
        return df.to_records(index=False)

    def history_deals_get(self, start=None, end=None, position=None):
        if position is not None:
            return tuple(d for d in self.deals if d.position_id == position)
        if start is not None:
            return tuple(d for d in self.deals if d.time >= start)
        return tuple(self.deals)

    def history_orders_get(self, start=None, end=None, position=None):
        if position is not None:
            return tuple(o for o in self.orders.values() if o.ticket == position)
        return tuple(self.orders.values())

    def positions_get(self, symbol=None):
        if symbol is not None:
            return tuple(p for p in self.positions.values() if p.symbol == symbol)
        return tuple(self.positions.values())

    def order_send(self, request):
        action = request.get("action", TRADE_ACTION_DEAL)
        if action == TRADE_ACTION_DEAL:
            return self._handle_deal(request)
        elif action == TRADE_ACTION_SLTP:
            return self._handle_sltp(request)
        elif action == TRADE_ACTION_PENDING:
            return self._handle_pending(request)
        return self._result(retcode=TRADE_RETCODE_FAIL)

    def order_delete(self, ticket):
        if ticket in self.orders:
            del self.orders[ticket]
            return self._result(retcode=TRADE_RETCODE_DONE)
        return self._result(retcode=TRADE_RETCODE_FAIL)

    # --- Internal state management ---

    def _alloc_ticket(self):
        ticket = self._next_ticket_val
        self._next_ticket_val += 1
        return ticket

    def _result(self, retcode=TRADE_RETCODE_DONE, order=0, price=0.0, volume=0.0):
        return type("OrderSendResult", (), {
            "retcode": retcode, "order": order, "price": price, "volume": volume,
        })()

    def _handle_deal(self, req):
        symbol = req["symbol"]
        volume = req["volume"]
        order_type = req["type"]
        price = req.get("price", self._tick.ask if order_type == ORDER_TYPE_BUY else self._tick.bid)
        position = req.get("position", 0)
        comment = req.get("comment", "TrendBot")
        magic = req.get("magic", 20240706)

        if position != 0:
            return self._close_position(position, volume, price, comment, magic)

        return self._open_position(symbol, order_type, volume, price, comment, magic)

    def _open_position(self, symbol, order_type, volume, price, comment, magic):
        self._now = int(datetime.now().timestamp())
        ticket = self._next_ticket_val
        self._next_ticket_val += 1
        pos = SimPosition(
            ticket=ticket, symbol=symbol, type=order_type,
            volume=volume, price_open=price, sl=0.0, tp=0.0,
            price_current=price, comment=comment, magic=magic,
            time=self._now,
        )
        self.positions[ticket] = pos
        return self._result(TRADE_RETCODE_DONE, ticket, price, volume)

    def _close_position(self, position_id, volume, price, comment, magic):
        pos = self.positions.get(position_id)
        if pos is None:
            return self._result(TRADE_RETCODE_FAIL)
        self._now = int(datetime.now().timestamp())
        remaining = pos.volume - volume
        if remaining <= 0:
            del self.positions[position_id]
        else:
            pos.volume = remaining
        is_buy = pos.type == ORDER_TYPE_BUY
        pnl = (price - pos.price_open) * volume * self._symbol_info.trade_tick_value if is_buy \
              else (pos.price_open - price) * volume * self._symbol_info.trade_tick_value
        deal = SimDeal(
            ticket=pos.ticket, order=pos.ticket, position_id=pos.ticket,
            symbol=pos.symbol, type=pos.type, price=price,
            volume=volume, profit=pnl, magic=magic,
            time=self._now, comment=comment,
        )
        self.deals.append(deal)
        self._account_info.balance += pnl
        self._account_info.equity += pnl
        self._account_info.profit = sum(
            p.volume * abs(p.price_current - p.price_open) * self._symbol_info.trade_tick_value
            for p in self.positions.values()
        )
        return self._result(TRADE_RETCODE_DONE, pos.ticket, price, volume)

    def _handle_sltp(self, req):
        position = req.get("position", 0)
        new_sl = req.get("sl")
        new_tp = req.get("tp")
        pos = self.positions.get(position)
        if pos is None:
            return self._result(TRADE_RETCODE_FAIL)
        if new_sl is not None:
            pos.sl = new_sl
        if new_tp is not None:
            pos.tp = new_tp
        return self._result(TRADE_RETCODE_DONE, pos.ticket, pos.price_open, pos.volume)

    def _handle_pending(self, req):
        ticket = self._alloc_ticket()
        order = SimOrder(
            ticket=ticket, symbol=req["symbol"],
            type=req["type"], volume=req["volume"],
            price=req.get("price", 0.0),
            sl=req.get("sl", 0.0), tp=req.get("tp", 0.0),
            comment=req.get("comment", "TrendBot-LIMIT"),
            time_done=self._now, time_setup=self._now,
        )
        self.orders[ticket] = order
        return self._result(TRADE_RETCODE_DONE, ticket, order.price, order.volume)

    # --- Sim control ---

    def set_price(self, bid, ask):
        self._tick.bid = bid
        self._tick.ask = ask
        for p in self.positions.values():
            mid = (bid + ask) / 2
            p.price_current = mid
            is_buy = p.type == ORDER_TYPE_BUY
            p.profit = (mid - p.price_open) * p.volume * self._symbol_info.trade_tick_value if is_buy \
                       else (p.price_open - mid) * p.volume * self._symbol_info.trade_tick_value
        self._account_info.profit = sum(p.profit for p in self.positions.values())
        self._account_info.equity = self._account_info.balance + self._account_info.profit

    def set_rate(self, symbol, timeframe, count, df):
        self._rates_cache[(symbol, timeframe)] = df

    def set_symbol_info(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._symbol_info, k, v)

    def set_account(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._account_info, k, v)
