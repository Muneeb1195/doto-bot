"""Config loading — settings.ini + credentials.ini + env vars."""

import configparser
import logging
import os

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
from state import CONFIG_DIR

SYMBOL_OVERRIDE_KEYS = {
    "ema_fast",
    "ema_slow",
    "ma_type",
    "atr_sl_mult",
    "rr",
    "risk_percent",
    "atr_period",
    "atr_sma_period",
    "max_positions_per_symbol",
    "adx_trend_threshold",
    "adx_range_threshold",
    "kelly_fraction",
    "max_risk_ratio",
    "volatility_min_ratio",
    "deviation",
    "mtf_h4_ema_period",
    "mtf_m15_ema_fast",
    "mtf_m15_ema_slow",
    "mtf_enabled",
}

SYMBOL_STRATEGY_MAP = {
    "ema_fast_period": int,
    "ema_slow_period": int,
    "ma_type": str,
    "atr_sl_multiplier": float,
    "atr_period": int,
    "atr_sma_period": int,
    "risk_reward_ratio": float,
    "risk_percent": float,
    "max_positions_per_symbol": int,
    "adx_trend_threshold": int,
    "kelly_fraction": float,
    "adx_range_threshold": int,
    "max_risk_ratio": float,
    "volatility_min_ratio": float,
    "deviation": int,
    "min_equity": float,
    "mr_rsi_oversold": int,
    "mr_rsi_overbought": int,
    "mtf_h4_ema_period": int,
    "mtf_m15_ema_fast": int,
    "mtf_m15_ema_slow": int,
    "mtf_enabled": lambda v: v.lower() == "true" if isinstance(v, str) else bool(v),
    "scoring_min_entry": float,
}

KEY_MAP = {
    "ema_fast_period": "ema_fast",
    "ema_slow_period": "ema_slow",
    "ma_type": "ma_type",
    "atr_sl_multiplier": "atr_sl_mult",
    "risk_reward_ratio": "rr",
    "risk_percent": "risk_percent",
    "atr_period": "atr_period",
    "atr_sma_period": "atr_sma_period",
    "max_positions_per_symbol": "max_positions_per_symbol",
    "adx_trend_threshold": "adx_trend_threshold",
    "adx_range_threshold": "adx_range_threshold",
    "kelly_fraction": "dr_kelly_fraction",
    "max_risk_ratio": "max_risk_ratio",
    "volatility_min_ratio": "volatility_min_ratio",
    "deviation": "deviation",
    "scoring_min_entry": "scoring_min_entry",
}


_TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10,
    "M12": 12, "M15": 15, "M20": 20, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240, "H6": 360, "H8": 480,
    "H12": 720, "D1": 1440, "W1": 10080, "MN1": 43200,
}

# MT5 timeframe enum constants (identical across all modern builds).
_TF_CONSTANTS = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, "M10": 10,
    "M12": 12, "M15": 15, "M20": 20, "M30": 30,
    "H1": 16385, "H2": 16386, "H3": 16387, "H4": 16388, "H6": 16390,
    "H8": 16392, "H12": 16396, "D1": 16408, "W1": 32769, "MN1": 49153,
}


def _resolve_timeframe(tf_name, ctx):
    """Resolve an MT5 timeframe by name, failing loud on an unknown value.

    Returns the MT5 enum constant (e.g. 16385 for H1) — this is what
    mt5.copy_rates_* expects. Uses a static mapping (these constants never
    change) so config load does not require a live MT5 connection — otherwise
    a cold-box boot would crash before the startup-grace retry loop can bring
    MT5 up.
    """
    tf = _TF_CONSTANTS.get(tf_name)
    if tf is None:
        raise ValueError(
            f"Invalid {ctx} timeframe '{tf_name}' — not in {_TF_CONSTANTS}. "
            f"Fix settings.ini instead of silently defaulting."
        )
    return tf


def validate_config(cfg):
    """Reject nonsensical parameter values at load time so the bot fails loud
    rather than sizing trades or setting stops from garbage config (M12)."""
    errors = []

    def _check(name, value, lo, hi, inclusive_lo=False, inclusive_hi=True):
        ok_lo = value >= lo if inclusive_lo else value > lo
        ok_hi = value <= hi if inclusive_hi else value < hi
        if not (ok_lo and ok_hi):
            lb = "[" if inclusive_lo else "("
            rb = "]" if inclusive_hi else ")"
            errors.append(f"{name}={value} out of range {lb}{lo}, {hi}{rb}")

    _check("risk_percent", cfg["risk_percent"], 0, 10)
    _check("daily_loss_pct", cfg["daily_loss_pct"], 0, 100)
    _check("max_positions", cfg["max_positions"], 0, 1000)
    _check("atr_period", cfg["atr_period"], 0, 10000)
    _check("atr_sl_mult", cfg["atr_sl_mult"], 0, 100)
    _check("rr", cfg["rr"], 0, 100)
    _check("dr_kelly_fraction", cfg["dr_kelly_fraction"], 0, 1)
    _check("scoring_min_entry", cfg["scoring_min_entry"], 0, 1, inclusive_lo=True)
    _check("ml_confidence", cfg["ml_confidence"], 0, 1, inclusive_lo=True)

    if cfg["ema_fast"] <= 0 or cfg["ema_slow"] <= 0:
        errors.append("ema_fast/ema_slow must be positive")
    elif cfg["ema_fast"] >= cfg["ema_slow"]:
        errors.append(f"ema_fast ({cfg['ema_fast']}) must be < ema_slow ({cfg['ema_slow']})")

    # Circuit breaker must trip at a deeper drawdown than the tail-risk cooldown,
    # otherwise the softer limit can never fire (mirrors the live/backtest order).
    if cfg["cb_dd_pct"] <= cfg["tr_max_dd_pct"]:
        errors.append(
            f"circuit_breaker_dd_pct ({cfg['cb_dd_pct']}) must be > max_portfolio_dd_pct ({cfg['tr_max_dd_pct']})"
        )

    for symbol, ov in cfg.get("symbol_strategy", {}).items():
        ef = ov.get("ema_fast_period")
        es = ov.get("ema_slow_period")
        if ef is not None and es is not None and ef >= es:
            errors.append(f"[{symbol}] ema_fast_period ({ef}) must be < ema_slow_period ({es})")

    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))


def load_config():
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")

    creds = configparser.ConfigParser()
    creds.read(CONFIG_DIR / "credentials.ini")

    try:
        cfg = {
            "symbol": settings.get("TRADING", "symbol", fallback="XAU500.raw"),
            "timeframe": _resolve_timeframe(settings.get("TRADING", "timeframe", fallback="H1"), "TRADING"),
            "risk_percent": float(settings.get("TRADING", "risk_percent", fallback=1.0)),
            "max_positions": int(settings.get("TRADING", "max_concurrent_positions", fallback=5)),
            "daily_loss_pct": float(settings.get("TRADING", "daily_loss_limit_percent", fallback=5.0)),
            "ema_fast": int(settings.get("STRATEGY", "ema_fast_period", fallback=50)),
            "ema_slow": int(settings.get("STRATEGY", "ema_slow_period", fallback=200)),
            "ma_type": settings.get("STRATEGY", "ma_type", fallback="kama"),
            "atr_period": int(settings.get("STRATEGY", "atr_period", fallback=14)),
            "atr_sl_mult": float(settings.get("STRATEGY", "atr_sl_multiplier", fallback=1.0)),
            "atr_sma_period": int(settings.get("STRATEGY", "atr_sma_period", fallback=20)),
            "rr": float(settings.get("STRATEGY", "risk_reward_ratio", fallback=2.0)),
            "htf_timeframe": settings.get("STRATEGY", "htf_timeframe", fallback="H4"),
            "htf_ema_fast": int(settings.get("STRATEGY", "htf_ema_fast_period", fallback=50)),
            "htf_ema_slow": int(settings.get("STRATEGY", "htf_ema_slow_period", fallback=200)),
            "htf_misalign_size_mult": float(settings.get("STRATEGY", "htf_misalign_size_mult", fallback=0.5)),
            "scale_out_enabled": settings.get("SCALE_OUT", "enabled", fallback="True").lower() == "true",
            "scale_out_close_fractions": [
                float(x.strip())
                for x in settings.get("SCALE_OUT", "close_fractions", fallback="0.20,0.20").split(",")
                if x.strip()
            ],
            "scale_out_tp_targets_atr": [
                float(x.strip())
                for x in settings.get("SCALE_OUT", "tp_targets_atr", fallback="1.5,2.5").split(",")
                if x.strip()
            ],
            "scale_out_tp_targets_rr": [
                float(x.strip())
                for x in settings.get("SCALE_OUT", "tp_targets_rr", fallback="0.50,0.75").split(",")
                if x.strip()
            ],
            "scale_out_breakeven_fraction": float(settings.get("SCALE_OUT", "breakeven_fraction", fallback="0.25")),
            "trade_journal": settings.get("LOGGING", "trade_journal", fallback="True").lower() == "true",
            "london_open": settings.get("SESSION", "london_open", fallback="13:00"),
            "london_close": settings.get("SESSION", "london_close", fallback="22:00"),
            "session_only": settings.get("SESSION", "trade_only_session", fallback="False").lower() == "true",
            "require_overlap": settings.get("SESSION", "require_overlap", fallback="False").lower() == "true",
            "skip_asian": settings.get("SESSION", "skip_asian", fallback="True").lower() == "true",
            "asian_open": settings.get("SESSION", "asian_open", fallback="05:00"),
            "asian_close": settings.get("SESSION", "asian_close", fallback="12:00"),
            "server": os.environ.get("MT5_SERVER") or creds.get("LOGIN", "server", fallback=""),
            "account": int(os.environ.get("MT5_ACCOUNT") or creds.get("LOGIN", "account", fallback="0")),
            "password": os.environ.get("MT5_PASSWORD") or creds.get("LOGIN", "password", fallback=""),
            "mt5_path": settings.get("MT5", "path", fallback="C:\\Program Files\\MetaTrader 5\\terminal64.exe"),
            "timeout_ms": int(settings.get("MT5", "timeout_ms", fallback=180000)),
            "call_timeout": int(settings.get("MT5", "call_timeout", fallback=30)),
            "adx_period": int(settings.get("ADX", "adx_period", fallback=14)),
            "adx_trend_threshold": int(settings.get("ADX", "adx_trend_threshold", fallback=25)),
            "adx_range_threshold": int(settings.get("ADX", "adx_range_threshold", fallback=20)),
            "fused_threshold": float(settings.get("FUSED_REGIME", "threshold", fallback=50.0)),
            "fused_buffer": float(settings.get("FUSED_REGIME", "buffer", fallback=5.0)),
            "max_cycle_seconds": int(settings.get("WATCHDOG", "max_cycle_seconds", fallback=180)),
            "reconnect_sleep": int(settings.get("WATCHDOG", "reconnect_sleep", fallback=30)),
            "cycle_sleep": int(settings.get("WATCHDOG", "cycle_sleep", fallback=10)),
            "verbose_debug": settings.get("WATCHDOG", "verbose_debug", fallback="false").lower() == "true",
            "mr_enabled": settings.get("MEAN_REVERSION", "mr_enabled", fallback="True").lower() == "true",
            "mr_timeframe": settings.get("MEAN_REVERSION", "mr_timeframe", fallback="M30"),
            "mr_rsi_period": int(settings.get("MEAN_REVERSION", "mr_rsi_period", fallback=14)),
            "mr_rsi_oversold": int(settings.get("MEAN_REVERSION", "mr_rsi_oversold", fallback=30)),
            "mr_rsi_overbought": int(settings.get("MEAN_REVERSION", "mr_rsi_overbought", fallback=70)),
            "mr_sl_atr_mult": float(settings.get("MEAN_REVERSION", "mr_sl_atr_mult", fallback=1.0)),
            "mr_tp_atr_mult": float(settings.get("MEAN_REVERSION", "mr_tp_atr_mult", fallback=1.5)),
            "mr_position_size_mult": float(settings.get("MEAN_REVERSION", "mr_position_size_mult", fallback=0.5)),
            "mr_htf_deviation": float(settings.get("MEAN_REVERSION", "mr_htf_deviation", fallback=0.0)),
            "mr_cooldown_enabled": settings.get("MEAN_REVERSION", "mr_cooldown_enabled", fallback="True").lower()
            == "true",
            "mr_cooldown_bars": int(settings.get("MEAN_REVERSION", "mr_cooldown_bars", fallback=2)),
            "vf_enabled": settings.get("VOLUME_FILTER", "enabled", fallback="True").lower() == "true",
            "vf_sma_period": int(settings.get("VOLUME_FILTER", "volume_sma_period", fallback=20)),
            "vf_kappa": float(settings.get("VOLUME_FILTER", "volume_kappa", fallback=1.2)),
            "vf_obv_enabled": settings.get("VOLUME_FILTER", "obv_divergence_enabled", fallback="True").lower()
            == "true",
            "vf_obv_lookback": int(settings.get("VOLUME_FILTER", "obv_lookback", fallback=20)),
            "ch_enabled": settings.get("CHANDELIER", "enabled", fallback="True").lower() == "true",
            "ch_atr_period": int(settings.get("CHANDELIER", "atr_period", fallback=14)),
            "ch_atr_mult": float(settings.get("CHANDELIER", "atr_multiplier", fallback=3.0)),
            "ch_atr_mult_partial": float(settings.get("CHANDELIER", "atr_multiplier_partial", fallback=1.5)),
            "ch_lookback": int(settings.get("CHANDELIER", "lookback_period", fallback=14)),
            "ch_mult_overrides": {},
            "ch_two_stage": settings.get("CHANDELIER", "two_stage_enabled", fallback="True").lower() == "true",
            "ch_loose_mult": float(settings.get("CHANDELIER", "loose_mult", fallback=3.5)),
            "ch_tight_mult": float(settings.get("CHANDELIER", "tight_mult", fallback=1.5)),
            "ch_two_stage_min_r": float(settings.get("CHANDELIER", "two_stage_min_r", fallback=3.0)),
            "ch_accelerate_enabled": settings.get("CHANDELIER", "accelerate_enabled", fallback="False").lower()
            == "true",
            "ch_accelerate_strength": float(settings.get("CHANDELIER", "accelerate_strength", fallback=0.20)),
            "ch_accelerate_period": int(settings.get("CHANDELIER", "accelerate_period", fallback=14)),
            "ch_accelerate_bars": int(settings.get("CHANDELIER", "accelerate_bars", fallback=5)),
            "dr_enabled": settings.get("DYNAMIC_RISK", "enabled", fallback="True").lower() == "true",
            "dr_kelly_fraction": float(settings.get("DYNAMIC_RISK", "kelly_fraction", fallback=0.25)),
            "dd_kelly_reduction_pct": float(settings.get("DYNAMIC_RISK", "dd_kelly_reduction_pct", fallback=5.0)),
            "dr_lookback": int(settings.get("DYNAMIC_RISK", "lookback_trades", fallback=50)),
            "dr_max_mult": float(settings.get("DYNAMIC_RISK", "max_risk_mult", fallback=1.5)),
            "dr_min_mult": float(settings.get("DYNAMIC_RISK", "min_risk_mult", fallback=0.25)),
            "dr_vol_adjust": settings.get("DYNAMIC_RISK", "volatility_adjust", fallback="True").lower() == "true",
            "max_risk_ratio": float(settings.get("DYNAMIC_RISK", "max_risk_ratio", fallback=2.0)),
            "scoring_enabled": settings.get("SCORING", "enabled", fallback="True").lower() == "true",
            "scoring_min_entry": float(settings.get("SCORING", "min_entry_score", fallback=0.55)),
            "scoring_confidence_bucket_high": float(settings.get("SCORING", "confidence_bucket_high", fallback=0.85)),
            "scoring_confidence_bucket_low": float(settings.get("SCORING", "confidence_bucket_low", fallback=0.60)),
            "scoring_high_conviction_mult": float(settings.get("SCORING", "high_conviction_mult", fallback=1.0)),
            "scoring_standard_edge_mult": float(settings.get("SCORING", "standard_edge_mult", fallback=0.85)),
            "scoring_low_conviction_mult": float(settings.get("SCORING", "low_conviction_mult", fallback=0.50)),
            "scoring_weights": {},
            "scoring_ml_fallback": float(settings.get("SCORING", "ml_fallback", fallback=0.60)),
            "max_tail_risk_pct": float(settings.get("DYNAMIC_RISK", "max_tail_risk_pct", fallback=1.5)),
            "volatility_min_ratio": float(settings.get("VOLATILITY_FILTER", "min_atr_ratio", fallback=0.5)),
            "adx_percentile_enabled": settings.get("ADX", "percentile_enabled", fallback="False").lower() == "true",
            "adx_percentile_window_days": int(settings.get("ADX", "percentile_window_days", fallback=180)),
            "ns_stale_ttl": int(settings.get("NEWS_SENTIMENT", "stale_ttl_minutes", fallback=120)),
            "ns_asset_class_aware": settings.get("NEWS_SENTIMENT", "asset_class_aware", fallback="True").lower()
            == "true",
            "be_enabled": settings.get("BREAKEVEN", "enabled", fallback="True").lower() == "true",
            "be_atr_mult": float(settings.get("BREAKEVEN", "atr_mult", fallback=1.0)),
            "max_hold_hours": int(settings.get("BREAKEVEN", "max_hold_hours", fallback=72)),
            "ml_enabled": settings.get("ML_SIGNAL", "enabled", fallback="True").lower() == "true",
            "ml_confidence": float(settings.get("ML_SIGNAL", "confidence_threshold", fallback=0.55)),
            "ml_model_path_template": settings.get("ML_SIGNAL", "model_path", fallback="models/model_{symbol}.pkl"),
            "ml_threshold_overrides": {},
            "ml_min_r": float(settings.get("ML_SIGNAL", "min_r", fallback=0.20)),
            "ml_max_r": float(settings.get("ML_SIGNAL", "max_r", fallback=2.0)),
            "ml_meta_threshold": float(settings.get("ML_SIGNAL", "meta_threshold", fallback=0.50)),
            "exhaustion_adx_threshold": int(settings.get("ADX", "exhaustion_adx_threshold", fallback=40)),
            "exhaustion_slope_threshold": float(settings.get("ADX", "exhaustion_slope_threshold", fallback=2.0)),
            "tape_bearish_pressure": float(settings.get("TAPE_READING", "bearish_pressure", fallback=0.35)),
            "tape_bullish_pressure": float(settings.get("TAPE_READING", "bullish_pressure", fallback=0.65)),
            "tape_range_ratio": float(settings.get("TAPE_READING", "range_ratio", fallback=1.2)),
            "spf_enabled": settings.get("SPREAD_FILTER", "enabled", fallback="True").lower() == "true",
            "spf_max_ratio": float(settings.get("SPREAD_FILTER", "max_spread_atr_ratio", fallback=0.30)),
            "mtf_enabled": settings.get("MTF", "enabled", fallback="False").lower() == "true",
            "mtf_agreement_threshold": float(settings.get("MTF", "agreement_threshold", fallback=0.5)),
            "mtf_h4_ema_period": int(settings.get("MTF", "h4_ema_period", fallback=100)),
            "le_enabled": settings.get("LIMIT_ENTRY", "enabled", fallback="True").lower() == "true",
            "le_score_threshold": float(settings.get("LIMIT_ENTRY", "score_threshold", fallback=0.80)),
            "le_max_cycles": int(settings.get("LIMIT_ENTRY", "max_cycles", fallback=30)),
            "le_offset_spreads": float(settings.get("LIMIT_ENTRY", "offset_spreads", fallback=0.5)),
            "pb_enabled": settings.get("TREND_ENTRY", "enabled", fallback="True").lower() == "true",
            "pb_atr_mult": float(settings.get("TREND_ENTRY", "pullback_atr_mult", fallback=0.5)),
            "pb_rsi_confirm": settings.get("TREND_ENTRY", "rsi_pullback_confirm", fallback="True").lower() == "true",
            "pb_volume_enabled": settings.get("TREND_ENTRY", "pb_volume_enabled", fallback="True").lower() == "true",
            "pb_volume_sma_period": int(settings.get("TREND_ENTRY", "pb_volume_sma_period", fallback=20)),
            "pb_volume_threshold": float(settings.get("TREND_ENTRY", "pb_volume_threshold", fallback=0.8)),
            "pb_confirm_bars": int(settings.get("TREND_ENTRY", "pb_confirm_bars", fallback=1)),
            "pb_structure_lookback": int(settings.get("TREND_ENTRY", "pb_structure_lookback", fallback=5)),
            "pb_atr_min_dist": float(settings.get("TREND_ENTRY", "pb_atr_min_dist", fallback=0.1)),
            "tr_enabled": settings.get("TAIL_RISK", "enabled", fallback="True").lower() == "true",
            "tr_sigma": float(settings.get("TAIL_RISK", "sigma_threshold", fallback=3.0)),
            "tr_max_dd_pct": float(settings.get("TAIL_RISK", "max_portfolio_dd_pct", fallback=8.0)),
            "tr_cooldown": int(settings.get("TAIL_RISK", "cooldown_minutes", fallback=60)),
            "tr_lookback": int(settings.get("TAIL_RISK", "lookback_bars", fallback=50)),
            "cb_enabled": settings.get("TAIL_RISK", "circuit_breaker_enabled", fallback="True").lower() == "true",
            "cb_dd_pct": float(settings.get("TAIL_RISK", "circuit_breaker_dd_pct", fallback=15.0)),
            "tape_enabled": settings.get("TAPE_READING", "enabled", fallback="True").lower() == "true",
            "tape_m1_lookback": int(settings.get("TAPE_READING", "m1_lookback", fallback=100)),
            "tape_imbalance_threshold": float(settings.get("TAPE_READING", "imbalance_threshold", fallback=0.20)),
            "eq_enabled": settings.get("EXECUTION_QUALITY", "enabled", fallback="True").lower() == "true",
            "eq_track_slippage": settings.get("EXECUTION_QUALITY", "track_slippage", fallback="True").lower() == "true",
            "eq_track_rejections": settings.get("EXECUTION_QUALITY", "track_rejections", fallback="True").lower()
            == "true",
            "ns_enabled": settings.get("NEWS_SENTIMENT", "enabled", fallback="False").lower() == "true",
            "ns_window_hours": int(settings.get("NEWS_SENTIMENT", "window_hours", fallback=6)),
            "ns_min_headlines": int(settings.get("NEWS_SENTIMENT", "min_headlines", fallback=3)),
            "ns_neg_threshold": float(settings.get("NEWS_SENTIMENT", "negative_threshold", fallback=0.60)),
            "ns_pos_threshold": float(settings.get("NEWS_SENTIMENT", "positive_threshold", fallback=0.60)),
            "symbols": [s.strip() for s in settings.get("PORTFOLIO", "symbols", fallback="XAU500.raw").split(",")],
            "max_total_positions": int(settings.get("PORTFOLIO", "max_total_positions", fallback=5)),
            "portfolio_risk_pct": float(settings.get("PORTFOLIO", "portfolio_risk_pct", fallback=3.0)),
            "max_open_risk_pct": float(settings.get("PORTFOLIO", "max_open_risk_pct", fallback=5.0)),
            "corr_enabled": settings.get("CORRELATION", "enabled", fallback="True").lower() == "true",
            "corr_lookback_hours": int(settings.get("CORRELATION", "lookback_hours", fallback=24)),
            "corr_reduction_max": float(settings.get("CORRELATION", "reduction_max", fallback=0.50)),
            "discord_url": os.environ.get("DISCORD_WEBHOOK_URL") or creds.get("WEBHOOK", "discord_url", fallback=""),
            "marketaux_api_key": os.environ.get("MARKETAUX_API_KEY") or creds.get("MARKETAUX", "api_key", fallback=""),
            "commission": float(settings.get("BACKTEST", "commission_per_lot", fallback=976.0)),
            "spread_model": float(settings.get("BACKTEST", "spread_model", fallback=1.0)),
            "deviation": int(settings.get("ORDER_EXECUTION", "deviation", fallback=50)),
            "magic": int(settings.get("ORDER_EXECUTION", "magic_number", fallback=20240706)),
            "mr_magic": int(settings.get("ORDER_EXECUTION", "mr_magic_number", fallback=20240707)),
        }
    except Exception as e:
        logging.error(f"Config load failed: {e}")
        raise

    try:
        raw_weights = settings.get("SCORING", "weights", fallback="")
        if raw_weights:
            cfg["scoring_weights"] = {}
            for item in raw_weights.split(","):
                if item and ":" in item:
                    k, v = item.split(":", 1)
                    cfg["scoring_weights"][k.strip()] = float(v.strip())
    except Exception as e:
        logging.warning(f"Failed to parse scoring weights: {e}")
        if not cfg["scoring_weights"]:
            cfg["scoring_weights"] = {"ml": 0.40, "spread": 0.30, "news": 0.30}

    try:
        raw_overrides = settings.get("CHANDELIER", "multiplier_overrides", fallback="")
        if raw_overrides:
            cfg["ch_mult_overrides"] = {}
            for item in raw_overrides.split(","):
                if item and ":" in item:
                    k, v = item.split(":", 1)
                    cfg["ch_mult_overrides"][k.strip()] = float(v.strip())
    except Exception as e:
        logging.warning(f"Failed to parse chandelier multiplier_overrides: {e}")
        cfg["ch_mult_overrides"] = {}

    try:
        raw_ml_overrides = settings.get("ML_SIGNAL", "threshold_overrides", fallback="")
        if raw_ml_overrides:
            cfg["ml_threshold_overrides"] = {}
            for item in raw_ml_overrides.split(","):
                if item and ":" in item:
                    k, v = item.split(":", 1)
                    cfg["ml_threshold_overrides"][k.strip()] = float(v.strip())
    except Exception as e:
        logging.warning(f"Failed to parse ML threshold_overrides: {e}")
        cfg["ml_threshold_overrides"] = {}

    cfg["symbol_strategy"] = {}
    for symbol in cfg["symbols"]:
        section = f"STRATEGY:{symbol}"
        if settings.has_section(section):
            overrides = {}
            for key, conv in SYMBOL_STRATEGY_MAP.items():
                try:
                    if settings.has_option(section, key):
                        overrides[key] = conv(settings.get(section, key))
                except Exception as e:
                    logging.warning(f"Failed to parse {section}/{key}: {e}")
            if overrides:
                cfg["symbol_strategy"][symbol] = overrides
    cfg["_global_strategy_defaults"] = {
        "ema_fast": cfg["ema_fast"],
        "ema_slow": cfg["ema_slow"],
        "ma_type": cfg.get("ma_type", "kama"),
        "atr_sl_mult": cfg["atr_sl_mult"],
        "rr": cfg["rr"],
        "risk_percent": cfg["risk_percent"],
        "adx_trend_threshold": cfg.get("adx_trend_threshold", 25),
        "adx_range_threshold": cfg.get("adx_range_threshold", 20),
        "kelly_fraction": cfg.get("dr_kelly_fraction", 0.25),
        "atr_period": cfg.get("atr_period", 14),
        "atr_sma_period": cfg.get("atr_sma_period", 20),
        "max_positions_per_symbol": cfg.get("max_positions_per_symbol", 1),
        "max_risk_ratio": cfg.get("max_risk_ratio", 2.0),
        "volatility_min_ratio": cfg.get("volatility_min_ratio", 0.5),
        "deviation": cfg.get("deviation", 50),
        "mtf_h4_ema_period": cfg.get("mtf_h4_ema_period", 100),
        "mtf_enabled": cfg.get("mtf_enabled", False),
    }

    for symbol in cfg["symbols"]:
        section = f"SCALE_OUT:{symbol}"
        if settings.has_section(section):
            so = {}
            try:
                if settings.has_option(section, "tp_targets_atr"):
                    so["tp_targets_atr"] = [
                        float(x.strip()) for x in settings.get(section, "tp_targets_atr").split(",") if x.strip()
                    ]
            except Exception:
                logging.warning("Failed to parse tp_targets_atr for %s", symbol, exc_info=True)
            try:
                if settings.has_option(section, "close_fractions"):
                    so["close_fractions"] = [
                        float(x.strip()) for x in settings.get(section, "close_fractions").split(",") if x.strip()
                    ]
            except Exception:
                logging.warning("Failed to parse close_fractions for %s", symbol, exc_info=True)
            if so:
                cfg.setdefault("symbol_scale_out", {})[symbol] = so

    for symbol in cfg["symbols"]:
        section = f"CHANDELIER:{symbol}"
        if settings.has_section(section):
            ch = {}
            try:
                if settings.has_option(section, "atr_multiplier"):
                    ch["atr_multiplier"] = float(settings.get(section, "atr_multiplier"))
            except Exception:
                logging.warning("Failed to parse atr_multiplier for %s", symbol, exc_info=True)
            try:
                if settings.has_option(section, "atr_multiplier_partial"):
                    ch["atr_multiplier_partial"] = float(settings.get(section, "atr_multiplier_partial"))
            except Exception:
                logging.warning("Failed to parse atr_multiplier_partial for %s", symbol, exc_info=True)
            if ch:
                cfg.setdefault("symbol_chandelier", {})[symbol] = ch

    # Fail loud on unknown htf/mr timeframe names (stored as strings, resolved
    # downstream) rather than letting them silently misbehave later (M12).
    _resolve_timeframe(cfg["htf_timeframe"], "STRATEGY htf_timeframe")
    _resolve_timeframe(cfg["mr_timeframe"], "MEAN_REVERSION mr_timeframe")

    validate_config(cfg)

    return cfg


def apply_symbol_strategy(cfg, symbol):
    globals_ = cfg.get("_global_strategy_defaults", {})
    for k in SYMBOL_OVERRIDE_KEYS:
        if k in globals_:
            cfg[k] = globals_[k]
    cfg["dr_kelly_fraction"] = cfg.get("kelly_fraction", 0.25)
    overrides = cfg.get("symbol_strategy", {}).get(symbol, {})
    if not overrides:
        return
    for ini_key, value in overrides.items():
        cfg_key = KEY_MAP.get(ini_key, ini_key)
        cfg[cfg_key] = value


def apply_symbol_overrides(cfg, symbol):
    so = cfg.get("symbol_scale_out", {}).get(symbol, {})
    if so:
        if "tp_targets_atr" in so:
            cfg["scale_out_tp_targets_atr"] = so["tp_targets_atr"]
        if "tp_targets_rr" in so:
            cfg["scale_out_tp_targets_rr"] = so["tp_targets_rr"]
        if "close_fractions" in so:
            cfg["scale_out_close_fractions"] = so["close_fractions"]
    ch = cfg.get("symbol_chandelier", {}).get(symbol, {})
    if ch:
        if "atr_multiplier" in ch:
            cfg["ch_atr_mult"] = ch["atr_multiplier"]
        if "atr_multiplier_partial" in ch:
            cfg["ch_atr_mult_partial"] = ch["atr_multiplier_partial"]


# save_bot_state / load_bot_state moved to state.py
# Keep forwarding references for backward compat:
