"""Tests for config.py — load_config, apply_symbol_strategy, apply_symbol_overrides."""

import sys

sys.path.insert(0, "bot")

import pytest

SETTINGS_INI = """[TRADING]
symbol = XAU500.raw
timeframe = H1
risk_percent = 1.0
max_concurrent_positions = 1
daily_loss_limit_percent = 5.0

[STRATEGY]
ema_fast_period = 50
ema_slow_period = 200
atr_period = 14
atr_sl_multiplier = 1.0
atr_sma_period = 20
risk_reward_ratio = 2.0
htf_timeframe = H4
htf_ema_fast_period = 50
htf_ema_slow_period = 200

[SCALE_OUT]
enabled = True
close_fractions = 0.30,0.30
tp_targets_atr = 1.5,2.5
breakeven_fraction = 0.0

[LOGGING]
trade_journal = True

[SESSION]
london_open = 13:00
london_close = 22:00
trade_only_session = False
require_overlap = False
skip_asian = True
asian_open = 05:00
asian_close = 12:00

[VOLUME_FILTER]
enabled = True
volume_sma_period = 20
volume_kappa = 1.2
obv_divergence_enabled = True
obv_lookback = 20

[CHANDELIER]
enabled = True
atr_period = 14
atr_multiplier = 2.5
lookback_period = 14
two_stage_enabled = True
two_stage_min_r = 3.0
loose_mult = 3.5
tight_mult = 1.5
accelerate_enabled = True
accelerate_strength = 0.20
accelerate_period = 14
accelerate_bars = 5

[WATCHDOG]
max_cycle_seconds = 180
reconnect_sleep = 30
cycle_sleep = 10
verbose_debug = false

[FINE_ENTRY]
enabled = True
m5_ema_period = 20
m5_pullback_atr_mult = 0.5
m5_rsi_min = 45
m5_rsi_max = 60
m5_cooldown_bars = 12
m5_breakout_mult = 1.5

[EXECUTION]
enabled = True
timeframe = M15
ema_fast_period = 9
ema_slow_period = 21
bias_timeout_hours = 6
bias_max_flips = 3

[SPREAD_FILTER]
enabled = True
max_spread_atr_ratio = 0.30

[DYNAMIC_RISK]
enabled = True
kelly_fraction = 0.25
lookback_trades = 50
max_risk_mult = 1.5
min_risk_mult = 0.25
volatility_adjust = True
max_risk_ratio = 2.0
max_tail_risk_pct = 1.5

[TREND_ENTRY]
enabled = False
pullback_atr_mult = 1.2
rsi_pullback_confirm = True

[VOLATILITY_FILTER]
min_atr_ratio = 0.5

[SCORING]
enabled = True
min_entry_score = 0.55
confidence_bucket_high = 0.85
confidence_bucket_low = 0.60
high_conviction_mult = 1.0
standard_edge_mult = 0.85
low_conviction_mult = 0.50
ml_fallback = 0.60
weights = exec:0.15,volume:0.10,volatility:0.10,spread:0.10,news:0.10,tape:0.10,ml:0.25,tail_risk:0.10

[ADX]
adx_period = 14
adx_trend_threshold = 25
adx_range_threshold = 20
percentile_enabled = True
percentile_window_days = 180
exhaustion_adx_threshold = 40
exhaustion_slope_threshold = 2.0

[ML_SIGNAL]
enabled = True
confidence_threshold = 0.50
meta_threshold = 0.50
model_path = models/model_{symbol}.pkl
threshold_overrides = BTCUSD.raw:0.65

[MEAN_REVERSION]
mr_enabled = True
mr_timeframe = M30
mr_rsi_period = 14
mr_rsi_oversold = 30
mr_rsi_overbought = 70
mr_sl_atr_mult = 1.0
mr_tp_atr_mult = 1.5
mr_position_size_mult = 0.5
mr_htf_deviation = 0.0

[TAPE_READING]
enabled = True
m1_lookback = 100
imbalance_threshold = 0.20
bearish_pressure = 0.35
bullish_pressure = 0.65
range_ratio = 1.2

[TAIL_RISK]
enabled = True
sigma_threshold = 3.0
max_portfolio_dd_pct = 8.0
cooldown_minutes = 60
lookback_bars = 50
circuit_breaker_enabled = True
circuit_breaker_dd_pct = 15.0

[EXECUTION_QUALITY]
enabled = True
track_slippage = True
track_rejections = True

[CORRELATION]
enabled = True
lookback_hours = 24
reduction_max = 0.50

[NEWS_SENTIMENT]
enabled = True
window_hours = 6
min_headlines = 1
negative_threshold = 0.40
positive_threshold = 0.40
stale_ttl_minutes = 120
asset_class_aware = True

[MT5]
path = C:\\Program Files\\MetaTrader 5\\terminal64.exe
timeout_ms = 180000
call_timeout = 30

[ORDER_EXECUTION]
deviation = 50
magic_number = 20240706

[PORTFOLIO]
symbols = XAU500.raw, BTCUSD.raw, NZDUSD.raw, US30.raw, GBPJPY.raw
max_total_positions = 5
portfolio_risk_pct = 3.0

[STRATEGY:XAU500.raw]
ema_fast_period = 8
ema_slow_period = 32
atr_sl_multiplier = 2.0
risk_reward_ratio = 2.0
adx_trend_threshold = 25
risk_percent = 1.0

[STRATEGY:NZDUSD.raw]
ema_fast_period = 10
ema_slow_period = 40
atr_sl_multiplier = 2.0
risk_reward_ratio = 1.5
adx_trend_threshold = 25
risk_percent = 1.0

[STRATEGY:BTCUSD.raw]
ema_fast_period = 3
ema_slow_period = 12
atr_sl_multiplier = 1.5
risk_reward_ratio = 2.0
adx_trend_threshold = 25
risk_percent = 1.0

[STRATEGY:US30.raw]
ema_fast_period = 25
ema_slow_period = 100
atr_sl_multiplier = 2.0
risk_reward_ratio = 2.0
adx_trend_threshold = 20
risk_percent = 1.0

[STRATEGY:GBPJPY.raw]
ema_fast_period = 6
ema_slow_period = 24
atr_sl_multiplier = 2.0
risk_reward_ratio = 2.5
adx_trend_threshold = 25
risk_percent = 1.0
"""

CREDENTIALS_INI = """[LOGIN]
account = 5128922
password = db9kB?Qg
server = DOTOGlobal-Real
"""


@pytest.fixture
def config_dir(tmp_path):
    """Create temp settings.ini and credentials.ini."""
    d = tmp_path / "config"
    d.mkdir()
    (d / "settings.ini").write_text(SETTINGS_INI)
    (d / "credentials.ini").write_text(CREDENTIALS_INI)
    return d


@pytest.fixture
def config_module(config_dir, monkeypatch):
    """Import config.py with patched paths and env vars."""
    monkeypatch.setattr("state.CONFIG_DIR", config_dir)
    monkeypatch.setattr("state.BASE_DIR", config_dir.parent)
    monkeypatch.setattr("state.STATE_FILE", config_dir.parent / "data" / "bot_state.json")
    monkeypatch.setenv("MT5_SERVER", "")
    monkeypatch.setenv("MT5_ACCOUNT", "")
    monkeypatch.setenv("MT5_PASSWORD", "")
    sys.modules.pop("config", None)
    import config as cfg_mod
    return cfg_mod


class TestLoadConfig:
    def test_loads_basic_keys(self, config_module):
        cfg = config_module.load_config()
        assert cfg["symbol"] == "XAU500.raw"
        assert cfg["risk_percent"] == 1.0
        assert cfg["max_positions"] == 1
        assert cfg["daily_loss_pct"] == 5.0
        assert cfg["ema_fast"] == 50
        assert cfg["ema_slow"] == 200
        assert cfg["atr_period"] == 14
        assert cfg["atr_sl_mult"] == 1.0

    def test_loads_scale_out(self, config_module):
        cfg = config_module.load_config()
        assert cfg["scale_out_enabled"] is True
        assert cfg["scale_out_close_fractions"] == [0.30, 0.30]
        assert cfg["scale_out_tp_targets_atr"] == [1.5, 2.5]

    def test_loads_session(self, config_module):
        cfg = config_module.load_config()
        assert cfg["london_open"] == "13:00"
        assert cfg["london_close"] == "22:00"
        assert cfg["skip_asian"] is True

    def test_loads_scoring(self, config_module):
        cfg = config_module.load_config()
        assert cfg["scoring_enabled"] is True
        assert cfg["scoring_min_entry"] == 0.55
        assert cfg["scoring_confidence_bucket_high"] == 0.85
        assert cfg["scoring_low_conviction_mult"] == 0.50
        w = cfg["scoring_weights"]
        assert isinstance(w, dict)
        assert w["exec"] == 0.15
        assert w["ml"] == 0.25
        assert w["tail_risk"] == 0.10

    def test_loads_portfolio(self, config_module):
        cfg = config_module.load_config()
        assert cfg["symbols"] == ["XAU500.raw", "BTCUSD.raw", "NZDUSD.raw", "US30.raw", "GBPJPY.raw"]
        assert cfg["max_total_positions"] == 5
        assert cfg["portfolio_risk_pct"] == 3.0

    def test_loads_symbol_strategy(self, config_module):
        cfg = config_module.load_config()
        ss = cfg["symbol_strategy"]
        xau = ss.get("XAU500.raw", {})
        assert xau["ema_fast_period"] == 8
        assert xau["ema_slow_period"] == 32
        assert xau["atr_sl_multiplier"] == 2.0
        assert xau["risk_percent"] == 1.0
        btc = ss.get("BTCUSD.raw", {})
        assert btc["ema_fast_period"] == 3
        assert btc["ema_slow_period"] == 12

    def test_loads_ml_config(self, config_module):
        cfg = config_module.load_config()
        assert cfg["ml_enabled"] is True
        assert cfg["ml_confidence"] == 0.50
        assert cfg["ml_threshold_overrides"] == {"BTCUSD.raw": 0.65}

    def test_loads_chandelier(self, config_module):
        cfg = config_module.load_config()
        assert cfg["ch_enabled"] is True
        assert cfg["ch_atr_mult"] == 2.5
        assert cfg["ch_two_stage"] is True

    def test_loads_correlation(self, config_module):
        cfg = config_module.load_config()
        assert cfg["corr_enabled"] is True
        assert cfg["corr_lookback_hours"] == 24
        assert cfg["corr_reduction_max"] == 0.50

    def test_loads_from_env_when_set(self, config_module, monkeypatch):
        monkeypatch.setenv("MT5_SERVER", "EnvServer-01")
        monkeypatch.setenv("MT5_ACCOUNT", "999999")
        monkeypatch.setenv("MT5_PASSWORD", "env_pass")
        import config as cfg_mod
        cfg = cfg_mod.load_config()
        assert cfg["server"] == "EnvServer-01"
        assert cfg["account"] == 999999
        assert cfg["password"] == "env_pass"

    def test_loads_from_creds_when_env_missing(self, config_module):
        import config as cfg_mod
        cfg = cfg_mod.load_config()
        assert cfg["server"] == "DOTOGlobal-Real"
        assert cfg["account"] == 5128922
        assert cfg["password"] == "db9kB?Qg"

    def test_fallback_keys_have_defaults(self, config_module):
        cfg = config_module.load_config()
        assert cfg["adx_trend_threshold"] == 25
        assert cfg["adx_range_threshold"] == 20
        assert cfg["exhaustion_adx_threshold"] == 40
        assert cfg["exhaustion_slope_threshold"] == 2.0
        assert cfg["call_timeout"] == 30
        assert cfg["deviation"] == 50
        assert cfg["magic"] == 20240706

    def test_loads_audit_fix_keys(self, config_module):
        cfg = config_module.load_config()
        # 4-gate: fused regime threshold and buffer
        assert cfg["fused_threshold"] == 50.0
        assert cfg["fused_buffer"] == 5.0
        # P2#18: verbose debug gating
        assert cfg["verbose_debug"] is False
        # P1#8: scale-out breakeven fraction
        assert cfg["scale_out_breakeven_fraction"] == 0.0
        # P0#2 / scoring ML fallback
        assert cfg["scoring_ml_fallback"] == 0.60
        # P0#2: ML meta-signal threshold
        assert cfg["ml_meta_threshold"] == 0.50
        # P1#4: mean-reversion HTF deviation
        assert cfg["mr_htf_deviation"] == 0.0

    def test_fused_regime_defaults(self, config_module):
        cfg = config_module.load_config()
        assert cfg["fused_threshold"] == 50.0
        assert cfg["fused_buffer"] == 5.0


class TestApplySymbolStrategy:
    def test_applies_xau_overrides(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "XAU500.raw")
        assert cfg["ema_fast"] == 8
        assert cfg["ema_slow"] == 32
        assert cfg["atr_sl_mult"] == 2.0
        assert cfg["rr"] == 2.0

    def test_applies_btc_overrides(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "BTCUSD.raw")
        assert cfg["ema_fast"] == 3
        assert cfg["ema_slow"] == 12
        assert cfg["atr_sl_mult"] == 1.5

    def test_applies_gbpjpy_overrides(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "GBPJPY.raw")
        assert cfg["ema_fast"] == 6
        assert cfg["rr"] == 2.5

    def test_resets_globals_between_symbols(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "XAU500.raw")
        assert cfg["ema_fast"] == 8
        config_module.apply_symbol_strategy(cfg, "NZDUSD.raw")
        assert cfg["ema_fast"] == 10

    def test_unknown_symbol_gets_globals(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "UNKNOWN.raw")
        assert cfg["ema_fast"] == 50
        assert cfg["ema_slow"] == 200

    def test_kelly_fraction_mapping(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "XAU500.raw")
        assert cfg["dr_kelly_fraction"] == cfg["kelly_fraction"]

    def test_atr_period_not_overridden(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_strategy(cfg, "XAU500.raw")
        assert cfg["atr_period"] == 14


class TestApplySymbolOverrides:
    def test_scale_out_overrides(self, config_module):
        cfg = config_module.load_config()
        cfg["symbol_scale_out"] = {
            "XAU500.raw": {"tp_targets_atr": [2.0, 4.0], "close_fractions": [0.50, 0.50]}
        }
        config_module.apply_symbol_overrides(cfg, "XAU500.raw")
        assert cfg["scale_out_tp_targets_atr"] == [2.0, 4.0]
        assert cfg["scale_out_close_fractions"] == [0.50, 0.50]

    def test_chandelier_overrides(self, config_module):
        cfg = config_module.load_config()
        cfg["symbol_chandelier"] = {
            "BTCUSD.raw": {"atr_multiplier": 3.0, "atr_multiplier_partial": 2.0}
        }
        config_module.apply_symbol_overrides(cfg, "BTCUSD.raw")
        assert cfg["ch_atr_mult"] == 3.0
        assert cfg["ch_atr_mult_partial"] == 2.0

    def test_no_overrides_uses_defaults(self, config_module):
        cfg = config_module.load_config()
        config_module.apply_symbol_overrides(cfg, "UNKNOWN.raw")
        assert cfg["scale_out_tp_targets_atr"] == [1.5, 2.5]
        assert cfg["scale_out_close_fractions"] == [0.30, 0.30]


class TestFailLoud:
    """Config must reject nonsense at load time instead of silently defaulting
    (agent audit C / M12)."""

    def test_resolve_timeframe_rejects_unknown(self, config_module, monkeypatch):
        import MetaTrader5 as mt5
        # Force the bogus name to be absent so _resolve_timeframe raises
        # (MagicMock would otherwise auto-vivify the attribute).
        monkeypatch.delattr(mt5, "TIMEFRAME_H11", raising=False)
        with pytest.raises(ValueError):
            config_module._resolve_timeframe("H11", "TRADING")
        # valid name resolves to the static minute value (no MT5 dependency)
        assert config_module._resolve_timeframe("H1", "TRADING") == 60
        assert config_module._resolve_timeframe("M15", "TRADING") == 15
        assert config_module._resolve_timeframe("D1", "TRADING") == 1440

    def test_validate_rejects_ema_ordering(self, config_module):
        cfg = config_module.load_config()
        cfg["ema_fast"] = 200
        cfg["ema_slow"] = 50
        with pytest.raises(ValueError):
            config_module.validate_config(cfg)

    def test_validate_rejects_risk_out_of_range(self, config_module):
        cfg = config_module.load_config()
        cfg["risk_percent"] = 500.0
        with pytest.raises(ValueError):
            config_module.validate_config(cfg)

    def test_validate_rejects_circuit_breaker_order(self, config_module):
        cfg = config_module.load_config()
        cfg["cb_dd_pct"] = 1.0
        cfg["tr_max_dd_pct"] = 8.0
        with pytest.raises(ValueError):
            config_module.validate_config(cfg)

    def test_validate_accepts_good_config(self, config_module):
        cfg = config_module.load_config()
        config_module.validate_config(cfg)  # must not raise


class TestClosedBars:
    """Explicit closed-bar helper (agent audit B)."""

    def test_drops_forming_bar(self):
        import pandas as pd
        from analytics import closed_bars
        df = pd.DataFrame({"close": [1, 2, 3, 4]})
        cb = closed_bars(df)
        assert len(cb) == 3
        assert list(cb["close"]) == [1, 2, 3]

    def test_passthrough_short_df(self):
        import pandas as pd
        from analytics import closed_bars
        df = pd.DataFrame({"close": [1]})
        assert closed_bars(df) is df

    def test_passthrough_none(self):
        from analytics import closed_bars
        assert closed_bars(None) is None

    def test_passthrough_empty(self):
        import pandas as pd
        from analytics import closed_bars
        df = pd.DataFrame({"close": []})
        assert len(closed_bars(df)) == 0
