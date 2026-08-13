"""Tests for execution.manage_positions — the exit tree, driven through the
injected Market seam (broker calls go through a fake market, not monkeypatched
mt5_call)."""

import sys  # noqa: E402
import time  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import execution  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import state as _st  # noqa: E402
from execution import manage_positions  # noqa: E402
from state import _chandelier_state, _scale_out_state  # noqa: E402


class Pos:
    def __init__(self, symbol="XAU500.raw", ticket=111, is_buy=True, price_open=2000.0,
                 price_current=2000.0, volume=0.1, sl=1990.0, tp=2010.0,
                 open_time=None, magic=None, comment="TrendBot"):
        self.symbol = symbol
        self.ticket = ticket
        self.type = execution.mt5.ORDER_TYPE_BUY if is_buy else execution.mt5.ORDER_TYPE_SELL
        self.price_open = price_open
        self.price_current = price_current
        self.volume = volume
        self.sl = sl
        self.tp = tp
        self.time = open_time if open_time is not None else time.time()
        self.magic = magic if magic is not None else 20240706
        self.comment = comment


class DoneResult:
    # Mirrors the real mt5.TradeResult: retcode/price only — NO profit field.
    # Realized P&L comes from the executed deal via realized_pnl().
    def __init__(self, price=0.0):
        self.retcode = execution.mt5.TRADE_RETCODE_DONE
        self.price = price
        self.order = 0


class Deal:
    # Mirrors an executed mt5.TradeDeal (the source of realized P&L).
    def __init__(self, profit=0.0):
        self.profit = profit


def _sinfo():
    class SInfo:
        point = 0.1
        trade_stops_level = 0
        trade_tick_value = 0.1
        trade_tick_size = 0.01
        volume_step = 0.01
        volume_min = 0.01
        volume_max = 100.0
    return SInfo()


class FakeMarket:
    """Deterministic stand-in for mt5_connect.Market; records every order_send."""

    def __init__(self, tick=None, sinfo=None, rates=None, send_results=None, deals=None):
        self.tick = tick
        self.sinfo = sinfo
        self.rates = rates
        self.send_results = list(send_results or [])
        self.deals = deals or {}  # {ticket: [Deal, ...]}
        self.orders = []
        self.sinfo_calls = 0
        self.tick_calls = 0
        self.rates_calls = 0

    def symbol_info_tick(self, symbol):
        self.tick_calls += 1
        return self.tick

    def symbol_info(self, symbol):
        self.sinfo_calls += 1
        return self.sinfo

    def get_rates(self, symbol, timeframe, count):
        self.rates_calls += 1
        return self.rates

    def order_send(self, request, _timeout=10):
        self.orders.append(request)
        if self.send_results:
            return self.send_results.pop(0)
        return None

    def get_deals(self, position):
        return self.deals.get(position, [])


@pytest.fixture
def pm_cfg(basic_cfg):
    cfg = dict(basic_cfg)
    cfg.update({
        "symbol": "XAU500.raw",
        "symbol_strategy": {},
        "mr_enabled": True,
        "mr_magic": 20240707,
        "magic": 20240706,
        "scale_out_enabled": True,
        "max_hold_hours": 72,
        "be_enabled": False,
        "mr_rsi_period": 14,
        "mr_timeframe": "M30",
    })
    return cfg


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # The close path sleeps 2s between positions; keep tests fast.
    monkeypatch.setattr(execution.time, "sleep", lambda s: None)


class TestNoPositions:
    def test_empty_book_returned_unchanged(self, pm_cfg):
        assert manage_positions(pm_cfg, "XAU500.raw", []) == []

    def test_other_symbol_positions_untouched(self, pm_cfg):
        other = Pos(symbol="BTCUSD.raw", ticket=999)
        assert manage_positions(pm_cfg, "XAU500.raw", [other]) == [other]


class TestClosePath:
    def _seed_scale_state(self, pm_cfg, pos):
        _scale_out_state[pos.ticket] = execution._init_scale_out_state(
            pm_cfg, pos.price_open, "buy", 50, _sinfo(),
            volume=pos.volume, atr_entry=2.0,
        )

    def _patch_side_effects(self, monkeypatch):
        closed = []
        monkeypatch.setattr(execution, "journal_close",
                            lambda ticket, px, pnl, pips, event: closed.append(
                                {"ticket": ticket, "pnl": pnl, "event": event}))
        saved = []
        monkeypatch.setattr(execution, "save_bot_state", lambda: saved.append(1))
        monkeypatch.setattr(execution, "get_filling_mode", lambda symbol: 0)
        return closed, saved

    def _old_position(self, **kw):
        return Pos(open_time=time.time() - 2 * 3600, **kw)

    def test_max_hold_close_sends_order_and_side_effects(self, pm_cfg, monkeypatch):
        pm_cfg["max_hold_hours"] = 1
        closed, saved = self._patch_side_effects(monkeypatch)
        pos = self._old_position()
        market = FakeMarket(
            tick=MagicMock(bid=1999.0, ask=1999.5),
            send_results=[DoneResult()],
            deals={pos.ticket: [Deal(profit=5.0)]},
        )
        self._seed_scale_state(pm_cfg, pos)
        _chandelier_state[pos.ticket] = {"ch_sl": 1900.0}
        try:
            book = manage_positions(
                pm_cfg, "XAU500.raw", [pos], atr=10.0,
                trend_signal="buy", regime="trending", market=market,
            )
        finally:
            _scale_out_state.pop(pos.ticket, None)
            _chandelier_state.pop(pos.ticket, None)
        assert book == []
        # Close order fields: opposite type, comment carries the reason.
        close_req = market.orders[-1]
        assert close_req["action"] == execution.mt5.TRADE_ACTION_DEAL
        assert close_req["type"] == execution.mt5.ORDER_TYPE_SELL
        assert close_req["position"] == pos.ticket
        assert close_req["comment"] == "TrendBot-MAX_HOLD"
        assert close_req["price"] == market.tick.bid
        # Post-close side effects.
        assert closed == [{"ticket": pos.ticket, "pnl": 5.0, "event": "MAX_HOLD"}]
        assert saved
        assert pos.ticket not in _scale_out_state
        assert pos.ticket not in _chandelier_state

    def test_close_failure_leaves_position_and_state(self, pm_cfg, monkeypatch):
        pm_cfg["max_hold_hours"] = 1
        closed, saved = self._patch_side_effects(monkeypatch)
        pos = self._old_position()
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5))  # order_send -> None
        self._seed_scale_state(pm_cfg, pos)
        try:
            book = manage_positions(
                pm_cfg, "XAU500.raw", [pos], atr=10.0, market=market,
            )
            state_survived = pos.ticket in _scale_out_state
        finally:
            _scale_out_state.pop(pos.ticket, None)
        assert book == [pos]
        assert closed == []
        assert not saved
        assert state_survived  # state untouched on failure

    def test_mr_loss_increments_streak(self, pm_cfg, monkeypatch):
        pm_cfg["max_hold_hours"] = 1
        self._patch_side_effects(monkeypatch)
        pos = self._old_position(magic=pm_cfg["mr_magic"], comment="TrendBot-MR")
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5),
                            send_results=[DoneResult()],
                            deals={pos.ticket: [Deal(profit=-3.0)]})
        self._seed_scale_state(pm_cfg, pos)
        try:
            manage_positions(pm_cfg, "XAU500.raw", [pos], atr=10.0, market=market)
            assert _st._mr_consecutive_losses["XAU500.raw"] == 1
            assert _st._mr_last_loss_time["XAU500.raw"] > 0
        finally:
            _scale_out_state.pop(pos.ticket, None)

    def test_mr_win_resets_streak(self, pm_cfg, monkeypatch):
        pm_cfg["max_hold_hours"] = 1
        self._patch_side_effects(monkeypatch)
        pos = self._old_position(magic=pm_cfg["mr_magic"], comment="TrendBot-MR")
        _st._mr_consecutive_losses["XAU500.raw"] = 2
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5),
                            send_results=[DoneResult()],
                            deals={pos.ticket: [Deal(profit=5.0)]})
        self._seed_scale_state(pm_cfg, pos)
        try:
            manage_positions(pm_cfg, "XAU500.raw", [pos], atr=10.0, market=market)
            assert _st._mr_consecutive_losses["XAU500.raw"] == 0
        finally:
            _scale_out_state.pop(pos.ticket, None)


class TestCloseDecision:
    def _patch_side_effects(self, monkeypatch):
        closed = []
        monkeypatch.setattr(execution, "journal_close",
                            lambda ticket, px, pnl, pips, event: closed.append(event))
        monkeypatch.setattr(execution, "save_bot_state", lambda: None)
        monkeypatch.setattr(execution, "get_filling_mode", lambda symbol: 0)
        return closed

    def test_reversal_close_without_sub_profit(self, pm_cfg, monkeypatch):
        # No rates -> get_current_atr returns None -> the 0.25-ATR profit guard
        # cannot protect the position -> counter-trend signal closes it.
        closed = self._patch_side_effects(monkeypatch)
        pos = Pos()
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5),
                            send_results=[DoneResult()],
                            deals={pos.ticket: [Deal(profit=1.0)]})
        try:
            book = manage_positions(
                pm_cfg, "XAU500.raw", [pos], trend_signal="sell",
                regime="trending", market=market,
            )
        finally:
            _scale_out_state.pop(pos.ticket, None)
        assert book == []
        assert market.orders[-1]["comment"] == "TrendBot-REVERSAL"
        assert closed == ["REVERSAL"]

    def test_reversal_blocked_by_sub_profit(self, pm_cfg, monkeypatch):
        closed = self._patch_side_effects(monkeypatch)
        n = 60
        closes = 100 + np.linspace(0, 3, n)
        rates = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "high": closes + 0.3,
            "low": closes - 0.3,
            "close": closes,
        })
        pos = Pos(price_current=2005.0)  # well beyond 0.25 * ATR in profit
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5), rates=rates)
        try:
            book = manage_positions(
                pm_cfg, "XAU500.raw", [pos], trend_signal="sell",
                regime="trending", market=market,
            )
        finally:
            _scale_out_state.pop(pos.ticket, None)
        assert book == [pos]  # protected — no close
        assert market.orders == []
        assert closed == []

    def test_mr_exit_wiring(self, pm_cfg, monkeypatch):
        # The MR exit DECISION is signals.check_mean_reversion_exit's own unit;
        # here we stub it True to verify the branch fires with the MR reason.
        closed = self._patch_side_effects(monkeypatch)
        monkeypatch.setattr(execution, "check_mean_reversion_exit",
                            lambda cfg, pos, market=None: True)
        pos = Pos(magic=pm_cfg["mr_magic"], comment="TrendBot-MR")
        market = FakeMarket(tick=MagicMock(bid=1999.0, ask=1999.5),
                            send_results=[DoneResult()],
                            deals={pos.ticket: [Deal(profit=2.0)]})
        try:
            book = manage_positions(
                pm_cfg, "XAU500.raw", [pos], regime="trending", market=market,
            )
        finally:
            _scale_out_state.pop(pos.ticket, None)
        assert book == []
        assert market.orders[-1]["comment"] == "TrendBot-MR_EXIT"
        assert closed == ["MR_EXIT"]


class TestTpRestoreAndWiring:
    def test_tp_restore_on_scale_out_position(self, pm_cfg):
        pos = Pos(tp=0.0)
        _scale_out_state[pos.ticket] = execution._init_scale_out_state(
            pm_cfg, pos.price_open, "buy", 50, _sinfo(),
            volume=pos.volume, atr_entry=2.0,
        )
        market = FakeMarket()
        try:
            manage_positions(pm_cfg, "XAU500.raw", [pos], atr=10.0, market=market)
        finally:
            _scale_out_state.pop(pos.ticket, None)
        assert len(market.orders) == 1
        req = market.orders[0]
        assert req["action"] == execution.mt5.TRADE_ACTION_SLTP
        assert req["tp"] == pytest.approx(pos.price_open + 50 * 0.1 * 2.0)

    def test_checks_receive_the_market(self, pm_cfg, monkeypatch):
        calls = {}
        for name in ("check_breakeven", "check_chandelier_exit", "check_scale_out"):
            calls[name] = []
            monkeypatch.setattr(
                execution, name,
                lambda cfg, pos, atr=None, market=None, _n=name: calls[_n].append(market),
            )
        pos = Pos()
        _scale_out_state[pos.ticket] = {"sl_points": 50, "point": 0.1, "rr": 2.0}
        market = FakeMarket()
        try:
            manage_positions(pm_cfg, "XAU500.raw", [pos], atr=10.0, market=market)
        finally:
            _scale_out_state.pop(pos.ticket, None)
        for name in calls:
            assert calls[name] == [market], f"{name} did not receive the market"

    def test_checks_route_broker_calls_through_market(self, pm_cfg):
        # check_breakeven / check_chandelier_exit must query the seam, not
        # mt5_call directly: a fake returning no symbol_info no-ops cleanly.
        pm_cfg["be_enabled"] = True
        pos = Pos()
        market = FakeMarket(sinfo=None)
        execution.check_breakeven(pm_cfg, pos, 10.0, market=market)
        execution.check_chandelier_exit(pm_cfg, pos, market=market)
        assert market.sinfo_calls == 2
        assert market.tick_calls == 0  # short-circuit on missing symbol_info
