"""Mock-based tests for filters.py — uses monkeypatch on mt5_call."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

from datetime import datetime  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import filters  # noqa: E402
import state as _st  # noqa: E402


class FakeAccountInfo:
    balance = 100000.0
    equity = 105000.0
    profit = 5000.0
    margin = 10000.0
    margin_free = 90000.0


class FakeTick:
    bid = 2000.0
    ask = 2000.5


class TestCheckMlSignal:
    def test_ml_disabled_returns_pass(self):
        passed, conf = filters.check_ml_signal({"ml_enabled": False, "symbol": "TEST"}, "buy")
        assert passed is True
        assert conf is None

    def test_no_model_returns_pass(self):
        passed, conf = filters.check_ml_signal({"ml_enabled": True, "symbol": "NONEXISTENT"}, "buy")
        assert passed is True
        assert conf is None

    def test_needs_250_bars_returns_pass(self, basic_cfg):
        basic_cfg["ml_enabled"] = True
        basic_cfg["symbol"] = "NONEXISTENT"
        passed, conf = filters.check_ml_signal(basic_cfg, "buy")
        assert passed is True
        assert conf is None


class TestCheckDailyLoss:
    def setup_method(self):
        _st._daily_realized_pnl = 0.0
        _st._daily_realized_date = datetime.now().date()
        _st._peak_balance = 0.0

    def _patch_datetime(self, monkeypatch):
        import datetime as dt  # noqa: E402
        class FakeDateTime:
            @classmethod
            def now(cls, tz=None): return dt.datetime(2020, 1, 2, 12, 0, 0, tzinfo=tz)
            @classmethod
            def date(cls, s=None): return dt.date(2020, 1, 2)
        monkeypatch.setattr(filters, "datetime", FakeDateTime)

    def test_no_loss_passes(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: FakeAccountInfo())
        monkeypatch.setattr(filters.mt5, "account_info", lambda: FakeAccountInfo())
        assert filters.check_daily_loss(basic_cfg) is True

    def test_small_loss_passes(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = -1000.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: FakeAccountInfo())
        assert filters.check_daily_loss(basic_cfg) is True

    def test_daily_loss_limit_hit(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = -6000.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: FakeAccountInfo())
        assert filters.check_daily_loss(basic_cfg) is False

    def test_no_account_info_returns_true(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: None)
        assert filters.check_daily_loss(basic_cfg) is True

    def test_zero_balance_returns_false(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        zero_acc = MagicMock()
        zero_acc.balance = 0.0
        zero_acc.profit = 0.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: zero_acc)
        assert filters.check_daily_loss(basic_cfg) is False

    def test_positive_pnl_resets_daily(self, basic_cfg, monkeypatch):
        self._patch_datetime(monkeypatch)
        _st._daily_realized_pnl = 5000.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: FakeAccountInfo())
        assert filters.check_daily_loss(basic_cfg) is True

    def test_loss_resets_on_new_day(self, basic_cfg, monkeypatch):
        import datetime as dt  # noqa: E402

        class FakeDateTime:
            @classmethod
            def now(cls, tz=None): return dt.datetime(2020, 1, 2, 12, 0, 0, tzinfo=tz)
            @classmethod
            def date(cls, s=None): return dt.date(2020, 1, 2)
        _st._daily_realized_date = dt.date(2020, 1, 1)
        _st._daily_realized_pnl = -6000.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=30, **kw: FakeAccountInfo())
        monkeypatch.setattr("filters.datetime", FakeDateTime)
        # Note: _daily_realized_pnl is now owned by journal.py; filters.py
        # does NOT reset it on day boundary. It treats stale values as 0.
        assert filters.check_daily_loss(basic_cfg) is True
        assert _st._daily_realized_pnl == -6000.0  # not zeroed by check_daily_loss

    def test_floating_pnl_does_not_trigger_halt(self, basic_cfg, monkeypatch):
        """P1#13: only realized PnL counts toward the daily-loss halt.
        Large open (floating) drawdown must NOT trip the 5% limit."""
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = 0.0  # no realized loss

        class FakePosition:
            profit = -9000.0  # large open drawdown on a 100k account

        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo()
            if "positions_get" in str(fn):
                return [FakePosition()]
            return None

        monkeypatch.setattr(filters, "mt5_call", fake_mt5_call)
        # If floating PnL were counted, -9000/100000 = -9% > 5% would halt.
        assert filters.check_daily_loss(basic_cfg) is True

    def test_realized_loss_triggers_halt_regardless_of_floating(self, basic_cfg, monkeypatch):
        """P1#13: realized loss is what halts, floating profit is ignored."""
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = -6000.0  # realized loss > 5%

        class FakePosition:
            profit = 4000.0  # open profit does not offset realized loss

        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo()
            if "positions_get" in str(fn):
                return [FakePosition()]
            return None

        monkeypatch.setattr(filters, "mt5_call", fake_mt5_call)
        assert filters.check_daily_loss(basic_cfg) is False

    def test_external_loss_triggers_halt(self, basic_cfg, monkeypatch):
        """The daily-loss halt must fire on an EXTERNAL (non-bot) trade that the
        journal never recorded. This is the 20k manual BTCUSD loss scenario:
        _daily_realized_pnl is 0 (bot saw nothing) but MT5 deal history shows
        a large realized loss."""
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = 0.0  # bot journal sees no loss

        class FakeDeal:
            time = 1577966400  # within the day
            position_id = 999  # non-zero so it counts as a trade, not a deposit
            profit = -6000.0
            commission = 0.0
            swap = 0.0

        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo()
            if "history_deals_get" in str(fn):
                return [FakeDeal()]
            return None

        monkeypatch.setattr(filters, "mt5_call", fake_mt5_call)
        assert filters.check_daily_loss(basic_cfg) is False

    def test_deals_unavailable_falls_back_to_journal(self, basic_cfg, monkeypatch):
        """When MT5 deal history is unavailable (returns None), the halt must
        still work off the in-journal realized PnL counter."""
        self._patch_datetime(monkeypatch)
        _st._daily_realized_date = datetime(2020, 1, 2).date()
        _st._daily_realized_pnl = -6000.0

        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo()
            if "history_deals_get" in str(fn):
                return None  # deal history unavailable
            return None

        monkeypatch.setattr(filters, "mt5_call", fake_mt5_call)
        assert filters.check_daily_loss(basic_cfg) is False


class TestCheckSpreadFilter:
    def test_disabled_passes(self, basic_cfg):
        basic_cfg["spf_enabled"] = False
        assert filters.check_spread_filter(basic_cfg) is True

    def test_good_spread_passes(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=5, **kw: FakeTick())
        monkeypatch.setattr(filters, "get_current_atr", lambda cfg: 50.0)
        assert filters.check_spread_filter(basic_cfg) is True

    def test_bad_spread_fails(self, basic_cfg, monkeypatch):
        bad_tick = MagicMock()
        bad_tick.bid = 2000.0
        bad_tick.ask = 2020.0
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=5, **kw: bad_tick)
        monkeypatch.setattr(filters, "get_current_atr", lambda cfg: 50.0)
        assert filters.check_spread_filter(basic_cfg) is False

    def test_no_tick_passes(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=5, **kw: None)
        assert filters.check_spread_filter(basic_cfg) is True

    def test_no_atr_passes(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(filters, "mt5_call", lambda fn, *a, _timeout=5, **kw: FakeTick())
        monkeypatch.setattr(filters, "get_current_atr", lambda cfg: None)
        assert filters.check_spread_filter(basic_cfg) is True
