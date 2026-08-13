"""Integration tests for full position lifecycles using Mt5Simulator.

Tests simulate realistic MT5 interactions across multiple cycles
to catch state corruption, unjournaled positions, and ordering bugs.
"""

import sys

sys.path.insert(0, "bot")

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tests.mock_mt5_lifecycle import ORDER_TYPE_BUY, ORDER_TYPE_SELL, Mt5Simulator

pytestmark = pytest.mark.usefixtures("mock_mt5")


@pytest.fixture
def mt5_sim():
    sim = Mt5Simulator()
    sim.set_symbol_info(point=0.01, trade_tick_value=0.1, trade_tick_size=0.01,
                        volume_step=0.01, volume_min=0.01, volume_max=100.0)
    sim.set_price(bid=2000.0, ask=2000.5)
    sim.set_account(balance=100000.0, equity=100000.0, profit=0.0)
    with patch("bot.mt5_connect.mt5_call", side_effect=lambda func, *a, **kw: func(*a, **kw)):
        with patch("bot.mt5_connect.mt5.order_delete", return_value=None):
            yield sim


@pytest.fixture
def basic_cfg():
    return {
        "symbol": "XAU500.raw",
        "timeframe": "H1",
        "magic": 20240706,
        "deviation": 50,
        "symbol_strategy": {"XAU500.raw": {}},
        "atr_period": 14,
        "atr_sl_mult": 2.0,
        "rr": 2.0,
        "trade_journal": True,
        "discord_url": None,
        "scale_out_enabled": True,
        "scale_out_close_fractions": [0.20, 0.20],
        "scale_out_tp_targets_rr": [0.50, 0.75],
        "scale_out_breakeven_fraction": 0.0,
        "ch_enabled": True,
        "ch_atr_mult": 3.0,
        "ch_atr_mult_partial": 2.0,
        "ch_two_stage": True,
        "ch_two_stage_min_r": 3.0,
        "ch_tight_mult": 1.5,
        "ch_loose_mult": 3.5,
        "ch_lookback": 50,
        "ch_atr_period": 14,
        "ch_accelerate_enabled": False,
        "le_enabled": False,
        "mr_sl_atr_mult": 1.0,
        "mr_tp_atr_mult": 1.0,
        "mr_enabled": False,
        "dr_enabled": False,
        "ml_enabled": False,
        "vf_enabled": False,
        "spf_enabled": False,
        "tape_enabled": False,
        "tr_enabled": False,
        "ns_enabled": False,
        "exec_enabled": False,
        "eq_enabled": False,
        "be_enabled": False,
        "risk_percent": 1.0,
        "max_tail_risk_pct": 5.0,
        "max_risk_ratio": 2.0,
        "scoring_enabled": False,
        "pb_enabled": False,
        "ma_type": "kama",
    }


@pytest.fixture(autouse=True)
def patch_mt5_module(mt5_sim):
    """Force re-import of bot modules so they pick up the simulator.

    Python caches ``import MetaTrader5 as mt5`` per module at import
    time.  Simply swapping ``sys.modules["MetaTrader5"]`` does NOT
    update existing references.  We delete bot modules from the cache
    so the first ``from bot.X import ...`` in each test re-imports
    them — at which point ``mt5`` resolves to the simulator.
    """
    _BOT_MODULES = [
        "execution", "mt5_connect", "journal", "filters",
        "risk", "regime", "ml_features", "signals",
    ]
    saved = {}
    for name in _BOT_MODULES:
        mod = sys.modules.pop(name, None)
        if mod is not None:
            saved[name] = mod
    sys.modules["MetaTrader5"] = mt5_sim
    # On Linux, mt5_connect instantiates mt5linux's client (mocked in conftest),
    # so its module-level ``mt5`` is a MagicMock — not the simulator.  Override it
    # so execution.check_scale_out / check_chandelier_exit route close requests
    # through the simulator instead of into a dead MagicMock.
    import bot.mt5_connect as _mt5c
    _mt5c.mt5 = mt5_sim
    # Consumers import the BARE module (tests add bot/ to sys.path), a distinct
    # object from bot.mt5_connect. Patch its `mt5` too so
    # mt5_connect.mt5_order_send (the single order-send source) routes through
    # the simulator instead of the dead MagicMock.
    import mt5_connect as _mt5c_bare
    _mt5c_bare.mt5 = mt5_sim
    try:
        yield
    finally:
        # Drop the re-imported bare modules and restore the pre-test originals
        # so later test files don't inherit a sim-tainted module. Restore is
        # unconditional: a module may have been re-imported during the test
        # (e.g. execution now imports signals at module level), and leaving
        # that fresh sim-bound copy in sys.modules would break subsequent
        # tests that patch the original object.
        for name, mod in saved.items():
            sys.modules[name] = mod
        sys.modules["MetaTrader5"] = MagicMock()


def _journal_rows():
    import csv

    import state as _st
    if not _st.TRADE_CSV.exists():
        return []
    with open(_st.TRADE_CSV, "r", newline="") as f:
        return list(csv.DictReader(f))


def _clear_state():
    import state as _st
    _st._scale_out_state.clear()
    _st._chandelier_state.clear()
    _st._exec_bias.clear()
    _st._last_trade_time.clear()
    _st._tail_risk_triggered.clear()
    _st._tail_risk_cooldown.clear()
    _st._circuit_breaker_triggered = False
    _st._peak_balance = 0.0
    _st._daily_realized_pnl = 0.0
    _st._daily_realized_date = datetime.now().date()
    _st._daily_loss_hit = False
    _st._pending_limits.clear()
    _st._rate_cache.clear()
    if _st.TRADE_CSV.exists():
        _st.TRADE_CSV.unlink()
    from journal import journal_init
    journal_init()


def _close_via_order_send(mt5_sim, cfg, pos, reason="REVERSAL"):
    """Helper: send a close deal for a position (mirrors main.py inline logic)."""
    from execution import mt5_order_send
    from journal import journal_close
    from mt5_connect import LIVE_MARKET, get_deviation, mt5_call, realized_pnl
    close_type = mt5_sim.ORDER_TYPE_SELL if pos.type == mt5_sim.ORDER_TYPE_BUY else mt5_sim.ORDER_TYPE_BUY
    tick = mt5_call(mt5_sim.symbol_info_tick, pos.symbol)
    if tick is None:
        return False
    price = tick.bid if close_type == mt5_sim.ORDER_TYPE_SELL else tick.ask
    sinfo = mt5_call(mt5_sim.symbol_info, pos.symbol)
    close_req = {
        "action": mt5_sim.TRADE_ACTION_DEAL, "symbol": pos.symbol,
        "volume": pos.volume, "type": close_type, "position": pos.ticket,
        "price": price, "deviation": get_deviation(cfg, pos.symbol),
        "magic": cfg.get("magic", 20240706), "comment": f"TrendBot-{reason}",
        "type_time": mt5_sim.ORDER_TIME_GTC, "type_filling": mt5_sim.ORDER_FILLING_IOC,
    }
    result = mt5_order_send(close_req)
    if result is not None and result.retcode == mt5_sim.TRADE_RETCODE_DONE:
        close_pnl = realized_pnl(LIVE_MARKET, pos.ticket)
        sp = sinfo.point if (sinfo and sinfo.point) else 0.001
        pips = abs(pos.price_open - price) / sp
        journal_close(pos.ticket, price, close_pnl, pips, reason)
        return True
    return False


def _set_rates_near_current(mt5_sim, symbol, tf, bars=200):
    """Set rates data with timestamps around now so chandelier/scale-out work."""
    now = pd.Timestamp.now()
    times = pd.date_range(now - pd.Timedelta(hours=bars - 1), periods=bars, freq="h")
    rng = np.random.default_rng(42)
    closes = 2000 + np.cumsum(rng.normal(0, 0.5, bars))
    df = pd.DataFrame({
        "time": times,
        "open": closes - rng.uniform(0, 0.5, bars),
        "high": closes + rng.uniform(0.1, 1.0, bars),
        "low": closes - rng.uniform(0.1, 1.0, bars),
        "close": closes,
        "tick_volume": rng.integers(100, 10000, bars),
        "spread": np.full(bars, 5),
    })
    mt5_sim.set_rate(symbol, tf, bars, df)


# ── Scenario 1: Open → SL/TP → close ──────────────────────────────────

class TestOpenModifyClose:
    def test_open_trend_trade(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.type == ORDER_TYPE_BUY
        rows = _journal_rows()
        opens = [r for r in rows if r["event"] == "OPEN"]
        assert len(opens) == 1
        assert opens[0]["ticket"] == str(pos.ticket)

    def test_open_mr_trade(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        result = _place_trade_inner(basic_cfg, "sell", atr=10.0, volume_mult=1.0, is_mr=True)
        assert result is True
        positions = mt5_sim.positions_get()
        assert len(positions) == 1
        assert positions[0].type == ORDER_TYPE_SELL

    def test_close_position_journals(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        assert len(positions) == 1
        pos = positions[0]
        closed = _close_via_order_send(mt5_sim, basic_cfg, pos, "REVERSAL")
        assert closed is True
        rows = _journal_rows()
        closes = [r for r in rows if r["event"] == "REVERSAL"]
        assert len(closes) == 1
        assert closes[0]["ticket"] == str(pos.ticket)

    def test_market_closed_skips_entry(self, mt5_sim, basic_cfg):
        from mt5_connect import market_open
        mt5_sim.set_symbol_info(trade_mode=999)
        tradeable = market_open("XAU500.raw")
        assert tradeable is False


# ── Scenario 2: Scale-out lifecycle ────────────────────────────────────

class TestScaleOutLifecycle:
    def test_open_with_scale_out_state(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner
        _clear_state()
        _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        positions = mt5_sim.positions_get()
        assert len(positions) == 1
        pos = positions[0]
        assert pos.ticket in _st._scale_out_state
        so = _st._scale_out_state[pos.ticket]
        assert so["step"] == 0
        assert so["num_partials"] == 2

    @pytest.mark.xfail(reason="MT5 simulator incomplete for scale-out lifecycle", strict=False)
    def test_scale_out_partial_close(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner, check_scale_out
        _clear_state()
        basic_cfg["rr"] = 2.0
        basic_cfg["atr_sl_mult"] = 2.0
        _set_rates_near_current(mt5_sim, basic_cfg["symbol"], mt5_sim.TIMEFRAME_H1, 200)
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[0]
        original_volume = pos.volume
        so = _st._scale_out_state[pos.ticket]
        tp_dist = so["sl_points"] * so["point"] * so.get("rr", 2.0)
        first_target = so["entry_price"] + tp_dist * 0.50
        mt5_sim.set_price(bid=first_target + 1, ask=first_target + 1.5)
        result = check_scale_out(basic_cfg, pos)
        assert result is True
        positions_after = mt5_sim.positions_get()
        assert len(positions_after) == 1
        assert positions_after[0].volume < original_volume

    @pytest.mark.xfail(reason="MT5 simulator incomplete for scale-out lifecycle", strict=False)
    def test_scale_out_full_close(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner, check_scale_out
        _clear_state()
        basic_cfg["rr"] = 2.0
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["scale_out_close_fractions"] = [1.0]
        basic_cfg["scale_out_tp_targets_rr"] = [0.75]
        _set_rates_near_current(mt5_sim, basic_cfg["symbol"], mt5_sim.TIMEFRAME_H1, 200)
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[0]
        so = _st._scale_out_state[pos.ticket]
        tp_dist = so["sl_points"] * so["point"] * so.get("rr", 2.0)
        target = so["entry_price"] + tp_dist * 0.75
        mt5_sim.set_price(bid=target + 2, ask=target + 2.5)
        result = check_scale_out(basic_cfg, pos)
        assert result is True
        positions_after = mt5_sim.positions_get()
        assert len(positions_after) == 0


# ── Scenario 3: Retry path ─────────────────────────────────────────────

class TestRetryPath:
    def test_retry_opens_without_sltp(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        assert len(positions) >= 1
        pos = positions[-1]
        assert pos.sl == 0.0
        assert pos.tp == 0.0

    def test_retry_journals_position(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        rows = _journal_rows()
        opens = [r for r in rows if r["event"] == "OPEN"]
        assert len(opens) >= 1

    def test_retry_then_close_journals_close(self, mt5_sim, basic_cfg):
        from execution import _place_trade_inner
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[-1]
        closed = _close_via_order_send(mt5_sim, basic_cfg, pos, "REVERSAL")
        assert closed is True
        rows = _journal_rows()
        closes = [r for r in rows if r["event"] == "REVERSAL"]
        assert len(closes) == 1


# ── Scenario 4: Chandelier exit ────────────────────────────────────────

class TestChandelierExit:
    def test_chandelier_trails_sl(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner, check_chandelier_exit
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        _set_rates_near_current(mt5_sim, basic_cfg["symbol"], mt5_sim.TIMEFRAME_H1, 200)
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[0]
        _st._chandelier_state[pos.ticket] = {"ch_sl": pos.sl}
        mt5_sim.set_price(bid=2010.0, ask=2010.5)
        check_chandelier_exit(basic_cfg, pos)
        updated = mt5_sim.positions_get()
        assert len(updated) == 1

    @pytest.mark.xfail(reason="MT5 simulator incomplete for chandelier exit", strict=False)
    def test_chandelier_breach_closes(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner, check_chandelier_exit
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        _set_rates_near_current(mt5_sim, basic_cfg["symbol"], mt5_sim.TIMEFRAME_H1, 200)
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[0]
        _st._chandelier_state[pos.ticket] = {"ch_sl": pos.sl}
        mt5_sim.set_price(bid=pos.sl - 5.0, ask=pos.sl - 4.5)
        check_chandelier_exit(basic_cfg, pos)
        positions_after = mt5_sim.positions_get()
        assert len(positions_after) == 0


# ── Scenario 5: State persistence (save/load/AATR) ─────────────────────

class TestStatePersistence:
    @pytest.fixture(autouse=True)
    def temp_state_dir(self, tmp_path):
        import state as _st
        _orig = _st.STATE_FILE, _st.TRADE_CSV, _st.BASE_DIR
        state_file = tmp_path / "bot_state.json"
        trade_csv = tmp_path / "trades.csv"
        _st.STATE_FILE = state_file
        _st.TRADE_CSV = trade_csv
        from journal import journal_init
        journal_init()
        yield
        _st.STATE_FILE, _st.TRADE_CSV, _st.BASE_DIR = _orig

    @pytest.fixture(autouse=True)
    def patch_state_save(self, mt5_sim, request):
        # Allow save_bot_state/load_bot_state to hit the real state module
        # by not patching mt5_call for state calls
        pass

    def test_save_and_load_roundtrip(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        assert len(positions) == 1
        _st.save_bot_state()
        saved = json.loads(_st.STATE_FILE.read_text())
        assert "scale_out_state" in saved
        _clear_state()
        _st.load_bot_state()
        assert len(_st._scale_out_state) == 1

    def test_empty_state_does_not_crash(self):
        import state as _st
        _clear_state()
        _st.load_bot_state()

    def test_corrupted_state_does_not_crash(self):
        import state as _st
        _st.STATE_FILE.write_text("{{garbage}}")
        _st.load_bot_state()

    def test_aatr_reconstruction_on_restart(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import _place_trade_inner
        _clear_state()
        basic_cfg["atr_sl_mult"] = 2.0
        basic_cfg["rr"] = 2.0
        result = _place_trade_inner(basic_cfg, "buy", atr=10.0, volume_mult=1.0)
        assert result is True
        positions = mt5_sim.positions_get()
        pos = positions[0]
        so = _st._scale_out_state[pos.ticket]
        so["atr_entry"] = 10.0
        _st.save_bot_state()
        _clear_state()
        _st.load_bot_state()
        restored = _st._scale_out_state.get(pos.ticket)
        assert restored is not None
        assert restored["atr_entry"] == 10.0


# ── Scenario 6: Limit order lifecycle ──────────────────────────────────

class TestLimitOrders:
    def test_place_and_cancel_limit(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import place_limit_order
        _clear_state()
        basic_cfg["le_enabled"] = True
        mt5_sim._pending_limits = _st._pending_limits
        result = place_limit_order(basic_cfg, "buy", atr=10.0, kelly_mult=1.0)
        assert result is not None

    def test_cancel_does_not_pop_on_fail(self, mt5_sim, basic_cfg):
        import state as _st
        from execution import cancel_pending_limit
        _st._pending_limits["XAU500.raw"] = {"ticket": 9999, "price": 1990.0, "signal": "buy"}
        # Correct arg order is (ticket, symbol). This mock's order_send returns
        # a failed retcode, so the order stays tracked (caller retries).
        result = cancel_pending_limit(9999, "XAU500.raw")
        assert result is False
        assert "XAU500.raw" in _st._pending_limits


# ── Scenario 7: Journal integrity ──────────────────────────────────────

class TestJournalIntegrity:
    @pytest.fixture(autouse=True)
    def temp_journal(self, tmp_path):
        import state as _st
        _orig = _st.TRADE_CSV
        _st.TRADE_CSV = tmp_path / "trades.csv"
        from journal import journal_init
        journal_init()
        yield
        _st.TRADE_CSV = _orig

    def test_open_close_pair(self, mt5_sim, basic_cfg):
        from journal import journal_close, journal_open
        journal_open(5001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 10.0)
        journal_close(5001, 2010.0, 100.0, 50.0, "CLOSE")
        rows = _journal_rows()
        opens = [r for r in rows if r["event"] == "OPEN"]
        closes = [r for r in rows if r["event"] == "CLOSE"]
        assert len(opens) == 1
        assert len(closes) == 1
        assert closes[0]["ticket"] == "5001"

    def test_no_orphan_after_reconcile(self, mt5_sim, basic_cfg):
        from journal import journal_open, reconcile_journal
        journal_open(6001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 10.0)
        reconcile_journal([])
        rows = _journal_rows()
        manual_closes = [r for r in rows if r["event"] == "MANUAL_CLOSE"]
        assert len(manual_closes) == 1
        assert manual_closes[0]["ticket"] == "6001"
