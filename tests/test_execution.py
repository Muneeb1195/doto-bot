"""Tests for execution.py — pure functions only (no MT5 dependency)."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import execution  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from execution import _init_scale_out_state, _place_trade_inner, log_execution_quality  # noqa: E402
from state import _exec_quality  # noqa: E402


class TestLogExecutionQuality:
    def setup_method(self):
        _exec_quality.clear()

    def test_tracks_slippage(self, basic_cfg):
        log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=100.1, rejected=False)
        assert "XAU500.raw" in _exec_quality
        assert _exec_quality["XAU500.raw"]["trades"] == 1
        assert _exec_quality["XAU500.raw"]["slippage_count"] == 1
        assert _exec_quality["XAU500.raw"]["slippage_sum"] > 0
        assert _exec_quality["XAU500.raw"]["rejections"] == 0

    def test_tracks_rejection(self, basic_cfg):
        log_execution_quality(basic_cfg, "BTCUSD.raw", 100.0, rejected=True)
        assert "BTCUSD.raw" in _exec_quality
        assert _exec_quality["BTCUSD.raw"]["rejections"] == 1
        assert _exec_quality["BTCUSD.raw"]["trades"] == 0

    def test_disabled_does_nothing(self, basic_cfg):
        basic_cfg["eq_enabled"] = False
        log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=100.1)
        assert "XAU500.raw" not in _exec_quality

    def test_multiple_trades_accumulate(self, basic_cfg):
        for i in range(5):
            log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=100.1)
        assert _exec_quality["XAU500.raw"]["trades"] == 5

    def test_zero_price_does_not_crash(self, basic_cfg):
        log_execution_quality(basic_cfg, "XAU500.raw", 0.0, fill_price=0.1)
        assert _exec_quality["XAU500.raw"]["trades"] == 1

    def test_none_fill_price_still_counts_trade(self, basic_cfg):
        log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=None)
        assert _exec_quality["XAU500.raw"]["trades"] == 1
        assert _exec_quality["XAU500.raw"]["slippage_count"] == 0

    def test_multiple_symbols_separate(self, basic_cfg):
        log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=100.1)
        log_execution_quality(basic_cfg, "BTCUSD.raw", 100.0, fill_price=99.9)
        assert len(_exec_quality) == 2

    def test_slippage_positive(self, basic_cfg):
        log_execution_quality(basic_cfg, "XAU500.raw", 100.0, fill_price=99.0)
        assert _exec_quality["XAU500.raw"]["slippage_sum"] > 0


class FakeSInfo:
    point = 0.1


class TestInitScaleOutState:
    def test_basic_init(self, basic_cfg):
        sinfo = FakeSInfo()
        result = _init_scale_out_state(
            basic_cfg, 100.0, "buy", 50, sinfo, is_mr=False, volume=0.1, atr_entry=2.0
        )
        assert result["step"] == 0
        assert result["entry_price"] == 100.0
        assert result["direction"] == "buy"
        assert result["sl_points"] == 50
        assert result["point"] == 0.1
        assert result["is_mr"] is False
        assert result["original_volume"] == 0.1
        assert result["atr_entry"] == 2.0
        assert result["close_fractions"] == [0.20, 0.20]
        assert result["tp_targets_rr"] == [0.50, 0.75]
        assert result["rr"] == 2.0
        assert result["num_partials"] == 2

    def test_sell_direction(self, basic_cfg):
        sinfo = FakeSInfo()
        result = _init_scale_out_state(basic_cfg, 100.0, "sell", 50, sinfo)
        assert result["direction"] == "sell"

    def test_mr_flag(self, basic_cfg):
        sinfo = FakeSInfo()
        result = _init_scale_out_state(basic_cfg, 100.0, "buy", 50, sinfo, is_mr=True)
        assert result["is_mr"] is True

    def test_custom_fractions_from_config(self, basic_cfg):
        basic_cfg["scale_out_close_fractions"] = [0.50, 0.30, 0.20]
        basic_cfg["scale_out_tp_targets_rr"] = [0.40, 0.60, 0.80]
        sinfo = FakeSInfo()
        result = _init_scale_out_state(basic_cfg, 100.0, "buy", 50, sinfo)
        assert result["close_fractions"] == [0.50, 0.30, 0.20]
        assert result["tp_targets_rr"] == [0.40, 0.60, 0.80]
        assert result["rr"] == 2.0
        assert result["num_partials"] == 3

    def test_default_sinfo_point(self):
        cfg = {"scale_out_close_fractions": [0.30, 0.30], "scale_out_tp_targets_atr": [1.5, 2.5]}
        sinfo = FakeSInfo()
        result = _init_scale_out_state(cfg, 100.0, "buy", 50, sinfo)
        assert result["point"] == 0.1


class FakeTick:
    ask = 2000.0
    bid = 1999.5


class FakeSymbolInfo:
    point = 0.01
    trade_stops_level = 0
    trade_tick_value = 0.1
    trade_tick_size = 0.01
    volume_step = 0.01
    volume_min = 0.01
    volume_max = 100.0


class FakeOrderResult:
    retcode = MagicMock()
    order = 12345
    price = 2000.0


class TestPlaceTradeSlFormula:
    """P0#1: SL distance must equal atr * sl_mult, not atr / sl_mult."""

    def _build_cfg(self, basic_cfg):
        cfg = dict(basic_cfg)
        cfg.update({
            "atr_sl_mult": 2.0,
            "rr": 2.0,
            "trade_journal": False,
            "discord_url": None,
            "scale_out_enabled": False,
            "regime": "trend",
            "magic": 20240706,
        })
        return cfg

    def _patch(self, monkeypatch, captured):
        tick = FakeTick()
        sinfo = FakeSymbolInfo()

        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            name = str(fn)
            if "symbol_info_tick" in name:
                return tick
            if "symbol_info" in name:
                return sinfo
            return None

        monkeypatch.setattr(execution, "mt5_call", fake_mt5_call)
        monkeypatch.setattr(execution, "calc_position_size", lambda c, p, sp, vm=1.0: 0.1)
        monkeypatch.setattr(execution, "get_deviation", lambda c, s: 50)
        monkeypatch.setattr(execution, "get_filling_mode", lambda s: 0)
        monkeypatch.setattr(execution, "_update_dynamic_deviation", lambda *a, **kw: None)
        monkeypatch.setattr(execution, "journal_open", lambda *a, **kw: None)
        monkeypatch.setattr(execution, "trade_open", lambda *a, **kw: None)
        monkeypatch.setattr(execution, "save_bot_state", lambda *a, **kw: None)

        res = FakeOrderResult()
        res.retcode = execution.mt5.TRADE_RETCODE_DONE

        def fake_send(req, _timeout=10):
            captured["req"] = req
            return res

        monkeypatch.setattr(execution, "mt5_order_send", fake_send)
        return tick

    def test_buy_sl_distance_equals_atr_times_sl_mult(self, basic_cfg, monkeypatch):
        captured = {}
        tick = self._patch(monkeypatch, captured)
        cfg = self._build_cfg(basic_cfg)
        ok = _place_trade_inner(cfg, "buy", atr=10.0, volume_mult=1.0)
        assert ok is True
        req = captured["req"]
        # SL distance in price must be atr * sl_mult = 10 * 2 = 20
        assert req["sl"] == pytest.approx(tick.ask - 20.0, abs=1e-6)
        # TP distance must be atr * sl_mult * rr = 10 * 2 * 2 = 40
        assert req["tp"] == pytest.approx(tick.ask + 40.0, abs=1e-6)

    def test_sell_sl_distance_equals_atr_times_sl_mult(self, basic_cfg, monkeypatch):
        captured = {}
        tick = self._patch(monkeypatch, captured)
        cfg = self._build_cfg(basic_cfg)
        ok = _place_trade_inner(cfg, "sell", atr=10.0, volume_mult=1.0)
        assert ok is True
        req = captured["req"]
        # For sell, price = bid; SL is above price by atr * sl_mult
        assert req["sl"] == pytest.approx(tick.bid + 20.0, abs=1e-6)
        assert req["tp"] == pytest.approx(tick.bid - 40.0, abs=1e-6)

    def test_mr_uses_mr_sl_mult(self, basic_cfg, monkeypatch):
        captured = {}
        tick = self._patch(monkeypatch, captured)
        cfg = self._build_cfg(basic_cfg)
        cfg["mr_sl_atr_mult"] = 1.0
        cfg["mr_tp_atr_mult"] = 1.5
        ok = _place_trade_inner(cfg, "buy", atr=10.0, volume_mult=1.0, is_mr=True)
        assert ok is True
        req = captured["req"]
        # MR SL distance = atr * mr_sl_atr_mult = 10 * 1 = 10
        assert req["sl"] == pytest.approx(tick.ask - 10.0, abs=1e-6)
        # MR TP distance = atr * mr_sl_atr_mult * mr_tp_atr_mult = 10 * 1 * 1.5 = 15
        assert req["tp"] == pytest.approx(tick.ask + 15.0, abs=1e-6)

    def test_trend_naked_close_is_journaled(self, basic_cfg, monkeypatch):
        # Drive _place_trade_inner into the naked-close path: initial order with
        # SL/TP fails, the no-SL/TP retry opens, the SL/TP modify fails, then the
        # position is closed naked. Agent audit M5: the journal must be paired
        # (previously trend left an orphan OPEN with no CLOSE).
        captured = {}
        self._patch(monkeypatch, captured)
        journal_calls = []
        monkeypatch.setattr(execution, "journal_close",
                            lambda ticket, px, pnl, pips, event: journal_calls.append((ticket, event)))
        SLTP = execution.mt5.TRADE_ACTION_SLTP
        res_done = FakeOrderResult()
        res_done.retcode = execution.mt5.TRADE_RETCODE_DONE
        res_done.order = 777
        res_fail = FakeOrderResult()
        res_fail.retcode = 999

        def fake_send(req, _timeout=10):
            if req.get("action") == SLTP:
                return res_fail            # SL/TP modify fails -> naked close
            if req.get("position") is not None:
                return res_done            # naked close (DEAL with position)
            if req.get("sl") not in (0, 0.0) or req.get("tp") not in (0, 0.0):
                return res_fail            # initial order with SL/TP fails
            return res_done                # no-SL/TP open succeeds

        monkeypatch.setattr(execution, "mt5_order_send", fake_send)
        cfg = self._build_cfg(basic_cfg)
        cfg["trade_journal"] = True
        ok = _place_trade_inner(cfg, "buy", atr=10.0, volume_mult=1.0, is_mr=False)
        assert ok is False
        events = [e for _, e in journal_calls]
        assert "TREND_NAKED_CLOSE" in events


class TestMinStopPoints:
    """Agent audit M1: stop distance floor must be instrument-scaled, not a
    hardcoded 50 points (which over-widened stops on small-point symbols)."""

    def test_no_fixed_50_floor_without_tick(self):
        sinfo = FakeSymbolInfo()
        # broker level 0 + modest spread buffer (10) -> 10, NOT 50.
        assert execution._min_stop_points(sinfo) == 10

    def test_spread_based_floor(self):
        sinfo = FakeSymbolInfo()
        tick = FakeTick()  # ask 2000, bid 1999.5 -> spread 50 points
        assert execution._min_stop_points(sinfo, tick) == 60

    def test_broker_level_respected(self):
        sinfo = FakeSymbolInfo()
        sinfo.trade_stops_level = 200
        tick = FakeTick()
        assert execution._min_stop_points(sinfo, tick) == 200


class TestThrottleAfterFill:
    """Agent audit M3: the entry throttle must be armed only after a successful
    fill, otherwise a rejected order blocks re-entry for the cooldown window."""

    def _key(self, basic_cfg, prefix):
        return f"{prefix}:{basic_cfg['symbol']}"

    def test_no_throttle_on_failed_order(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(execution, "_place_trade_inner", lambda *a, **k: False)
        key = self._key(basic_cfg, "trend")
        execution._last_trade_time.pop(key, None)
        assert execution.place_trade(basic_cfg, "buy", 1.0) is False
        assert key not in execution._last_trade_time

    def test_throttle_set_on_success(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(execution, "_place_trade_inner", lambda *a, **k: True)
        key = self._key(basic_cfg, "trend")
        execution._last_trade_time.pop(key, None)
        assert execution.place_trade(basic_cfg, "buy", 1.0) is True
        assert key in execution._last_trade_time

    def test_mr_throttle_only_on_success(self, basic_cfg, monkeypatch):
        basic_cfg["mr_position_size_mult"] = 1.0
        monkeypatch.setattr(execution, "_place_trade_inner", lambda *a, **k: False)
        key = self._key(basic_cfg, "mr")
        execution._last_trade_time.pop(key, None)
        assert execution.place_mean_reversion_trade(basic_cfg, "buy", 1.0) is False
        assert key not in execution._last_trade_time


class TestScaleOutMinLot:
    """Agent audit M2: a sub-minimum-lot partial must NOT liquidate the entire
    position. When the partial rounds below the broker minimum lot (or would
    cover the whole remaining position on a non-final step) the step is skipped
    and advanced instead of closing everything."""

    def _patch(self, monkeypatch, captured, volume_min=0.05):
        class Tick:
            ask = 2500.0
            bid = 2499.5

        class SInfo:
            point = 0.01
            trade_stops_level = 0
            trade_tick_value = 0.1
            trade_tick_size = 0.01
            volume_step = 0.01
            volume_min = 0.05
            volume_max = 100.0

        SInfo.volume_min = volume_min
        res = FakeOrderResult()
        res.retcode = execution.mt5.TRADE_RETCODE_DONE

        # check_scale_out routes all broker access through the Market seam;
        # drive it with a deterministic fake instead of patching mt5_call.
        class Market:
            def symbol_info_tick(self, symbol):
                return Tick()

            def symbol_info(self, symbol):
                return SInfo()

            def get_rates(self, symbol, timeframe, count):
                return pd.DataFrame({"close": np.ones(30)})

            def order_send(self, req, _timeout=10):
                captured.setdefault("calls", 0)
                captured["calls"] += 1
                captured.setdefault("reqs", []).append(req)
                return res

        monkeypatch.setattr(execution, "calc_atr", lambda *a, **kw: 2.0)
        monkeypatch.setattr(execution, "get_deviation", lambda c, s: 50)
        monkeypatch.setattr(execution, "get_filling_mode", lambda s: execution.mt5.ORDER_FILLING_FOK)
        monkeypatch.setattr(execution, "journal_close", lambda *a, **kw: None)
        monkeypatch.setattr(execution, "trade_partial", lambda *a, **kw: None)
        monkeypatch.setattr(execution, "save_bot_state", lambda *a, **kw: None)
        return Market(), SInfo()

    def _make_state_and_position(self, cfg, volume=0.1):
        sinfo = FakeSymbolInfo()
        state = _init_scale_out_state(cfg, 2000.0, "buy", 50, sinfo,
                                      is_mr=False, volume=volume, atr_entry=2.0)

        class Pos:
            symbol = cfg["symbol"]
            ticket = 999
            type = execution.mt5.ORDER_TYPE_BUY
            price_open = 2000.0
            sl = 1900.0
            tp = 2200.0

        Pos.volume = volume
        return state, Pos()

    def test_sub_min_lot_partial_does_not_liquidate(self, basic_cfg, monkeypatch):
        # 40% partial of 0.1 lot = 0.04 < volume_min 0.05. Old code set
        # close_vol = position.volume (full liquidation). Fix must bump to the
        # minimum tradable lot and close only that (remaining 0.05 stays open).
        captured = {}
        basic_cfg["volume_min"] = 0.05
        market, _ = self._patch(monkeypatch, captured, volume_min=0.05)
        state, pos = self._make_state_and_position(basic_cfg, volume=0.1)
        execution._scale_out_state[pos.ticket] = state
        try:
            ok = execution.check_scale_out(basic_cfg, pos, market=market)
            assert ok is True
            close_reqs = [r for r in captured.get("reqs", [])
                          if r.get("action") == execution.mt5.TRADE_ACTION_DEAL]
            assert len(close_reqs) == 1
            assert close_reqs[0]["volume"] == pytest.approx(0.05, abs=1e-9)
            assert close_reqs[0]["volume"] < pos.volume
        finally:
            execution._scale_out_state.pop(pos.ticket, None)

    def test_sub_min_lot_skip_when_partial_exceeds_remaining(self, basic_cfg, monkeypatch):
        # On a non-final scale-out step, the bumped min-lot close would cover the
        # entire remaining position. That must SKIP (no order) and advance the
        # step, never liquidate everything early.
        captured = {}
        basic_cfg["volume_min"] = 0.1
        market, _ = self._patch(monkeypatch, captured, volume_min=0.1)
        state, pos = self._make_state_and_position(basic_cfg, volume=0.1)
        execution._scale_out_state[pos.ticket] = state
        try:
            ok = execution.check_scale_out(basic_cfg, pos, market=market)
            assert ok is False
            assert captured.get("calls", 0) == 0
            assert execution._scale_out_state[pos.ticket]["step"] == 1
        finally:
            execution._scale_out_state.pop(pos.ticket, None)


class TestCancelPendingLimit:
    def _patch_send(self, monkeypatch, captured):
        import state as _st

        res = MagicMock()
        res.retcode = execution.mt5.TRADE_RETCODE_DONE

        def fake_send(req, _timeout=5):
            captured["req"] = req
            return res

        monkeypatch.setattr(execution, "mt5_order_send", fake_send)
        _st._pending_limits.clear()
        return _st, res

    def test_sends_trade_action_remove(self, monkeypatch):
        # Regression: cancel must use order_send + TRADE_ACTION_REMOVE, NOT the
        # non-existent mt5.order_delete (which silently failed and left the order live).
        captured = {}
        _st, res = self._patch_send(monkeypatch, captured)
        _st._pending_limits["XAU500.raw"] = {"ticket": 9999, "price": 1990.0, "signal": "buy"}
        ok = execution.cancel_pending_limit(9999, "XAU500.raw")
        assert ok is True
        assert captured["req"]["action"] == execution.mt5.TRADE_ACTION_REMOVE
        assert captured["req"]["order"] == 9999
        assert "XAU500.raw" not in _st._pending_limits

    def test_pops_tracking_on_success(self, monkeypatch):
        captured = {}
        _st, res = self._patch_send(monkeypatch, captured)
        _st._pending_limits["XAU500.raw"] = {"ticket": 9999}
        ok = execution.cancel_pending_limit(9999, "XAU500.raw")
        assert ok is True
        assert "XAU500.raw" not in _st._pending_limits

    def test_keeps_tracking_on_failure(self, monkeypatch):
        captured = {}
        _st, res = self._patch_send(monkeypatch, captured)
        res.retcode = 10027  # TRADE_RETCODE_TRADE_DISABLED
        _st._pending_limits["XAU500.raw"] = {"ticket": 9999}
        ok = execution.cancel_pending_limit(9999, "XAU500.raw")
        assert ok is False
        assert "XAU500.raw" in _st._pending_limits
