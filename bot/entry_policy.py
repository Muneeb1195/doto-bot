"""Entry policy — the 4-gate pipeline extracted from main.py.

Deep module that owns the ordering Gate1 fused regime → Gate2 signal/MR/MTF →
HTF 3-state → Gate3 ML → Gate4 execution sanity → MR cooldown. Main.py does
only I/O (positions, risk guards, sizing) and delegates entry evaluation here,
so the whole gate order is testable without the 10 s loop.

The interface is the test surface — callers supply an already-built sym_cfg
and receive a value-type outcome. No global state is mutated except the
RegimeGate hysteresis (via state._regime_gate_state) which is owned by
signals._get_regime_gate and deliberately stateful.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import state as _st
from analytics import fused_regime_score
from filters import check_ml_gate, check_spread_filter, check_tape_reading, check_volume_filter
from mt5_connect import get_rates
from regime import get_current_atr
from signals import _get_regime_gate, check_htf_trend, get_mean_reversion_signal, get_mtf_fused_signal, get_signal


@dataclass(frozen=True)
class EntryOutcome:
    signal_entry: str | None
    entry_atr: float | None
    atr: float | None
    entry_type: str | None
    mtf_confidence: float | None
    gate_open: bool
    regime: str
    fused_score: float | None
    trend_signal: str | None
    mr_signal: str | None
    # Filter outcome
    blocked: bool
    block_reason: str | None  # regime_gate|no_signal|htf_block|ml_gate|sanity|mr_cooldown|htf_soft (soft still allows) ?
    htf_size_mult: float
    confidence_mult: float | None
    ml_conf: float | None


def evaluate_signal_gate(sym_cfg, market=None) -> tuple:
    """Gate1+Gate2 only — returns signal tuple without filter checks."""
    symbol = sym_cfg["symbol"]
    gate = _get_regime_gate(symbol, sym_cfg)
    G1_BARS = 100
    # Prefer market seam when provided, else global get_rates (seam half-adopted).
    if market is not None and hasattr(market, "get_rates"):
        df_g1 = market.get_rates(symbol, sym_cfg["timeframe"], G1_BARS)
    else:
        df_g1 = get_rates(symbol, sym_cfg["timeframe"], G1_BARS)
    fused_score = fused_regime_score(
        df_g1.iloc[:-1] if df_g1 is not None and len(df_g1) > 1 else df_g1, sym_cfg
    )
    gate_open = gate.update(fused_score)

    trend_signal = None
    mr_signal = None
    trend_atr = None
    mr_atr = None
    entry_type = None
    mtf_confidence = None

    if gate_open:
        if sym_cfg.get("mtf_enabled", False):
            trend_signal, trend_atr, entry_type, mtf_confidence = get_mtf_fused_signal(sym_cfg)
        else:
            trend_signal, trend_atr, entry_type = get_signal(sym_cfg)
    else:
        # Chop regime — MR first, then fallback to trend/MTF
        if sym_cfg.get("mr_enabled") and len(_st._pending_limits) == 0:  # caller will also check per-symbol count; keep parity: check inside main uses len(positions_sym)==0
            # We cannot know positions_sym len here without arg; caller passes via wrapper below.
            # Keep MR attempt unconditional — wrapper will gate on positions_sym len.
            mr_signal, mr_atr = get_mean_reversion_signal(sym_cfg)
        # Fallback will be handled by the full evaluate below that knows positions_sym

    # Note: fallback MTF/single after MR is handled in evaluate_entry which knows positions_sym
    atr = trend_atr or mr_atr
    if atr is None:
        atr = get_current_atr(sym_cfg, market=market) if market is not None else get_current_atr(sym_cfg)

    signal_entry = mr_signal if (mr_signal is not None and not gate_open) else trend_signal
    entry_atr = mr_atr if (mr_signal is not None and not gate_open) else trend_atr
    if entry_atr is None:
        entry_atr = atr
    regime = "ranging" if not gate_open else "trending"
    return (
        signal_entry,
        entry_atr,
        atr,
        entry_type,
        mtf_confidence,
        gate_open,
        regime,
        fused_score,
        trend_signal,
        mr_signal,
    )


def evaluate_entry(sym_cfg, market=None, positions_sym=None) -> EntryOutcome:
    """Full gate chain G1→G2→HTF→G3→G4→MR-cooldown.

    `positions_sym` is the per-symbol position list (for MR gate len==0 check);
    pass None to skip that guard (useful in tests).
    """
    symbol = sym_cfg["symbol"]
    gate = _get_regime_gate(symbol, sym_cfg)
    G1_BARS = 100
    if market is not None and hasattr(market, "get_rates"):
        df_g1 = market.get_rates(symbol, sym_cfg["timeframe"], G1_BARS)
    else:
        df_g1 = get_rates(symbol, sym_cfg["timeframe"], G1_BARS)
    fused_score = fused_regime_score(
        df_g1.iloc[:-1] if df_g1 is not None and len(df_g1) > 1 else df_g1, sym_cfg
    )
    gate_open = gate.update(fused_score)

    trend_signal = None
    mr_signal = None
    trend_atr = None
    mr_atr = None
    entry_type = None
    mtf_confidence = None

    if gate_open:
        if sym_cfg.get("mtf_enabled", False):
            trend_signal, trend_atr, entry_type, mtf_confidence = get_mtf_fused_signal(sym_cfg)
        else:
            trend_signal, trend_atr, entry_type = get_signal(sym_cfg)
    else:
        # Chop: MR only if flat book
        flat = positions_sym is not None and len(positions_sym) == 0
        # When caller omits positions_sym we still attempt MR (conservative)
        if sym_cfg.get("mr_enabled") and (positions_sym is None or flat):
            mr_signal, mr_atr = get_mean_reversion_signal(sym_cfg)
        if mr_signal is None:
            if sym_cfg.get("mtf_enabled", False):
                trend_signal, trend_atr, entry_type, mtf_confidence = get_mtf_fused_signal(sym_cfg)
            else:
                trend_signal, trend_atr, entry_type = get_signal(sym_cfg)

    atr = trend_atr or mr_atr
    if atr is None:
        try:
            atr = get_current_atr(sym_cfg, market=market) if market is not None else get_current_atr(sym_cfg)
        except TypeError:
            atr = get_current_atr(sym_cfg)

    signal_entry = mr_signal if (mr_signal is not None and not gate_open) else trend_signal
    entry_atr = mr_atr if (mr_signal is not None and not gate_open) else trend_atr
    if entry_atr is None:
        entry_atr = atr
    regime = "ranging" if not gate_open else "trending"

    # --- No signal → blocked (caller maps to regime_gate vs no_signal) ---
    if signal_entry is None:
        reason = "regime_gate" if not gate_open else "no_signal"
        return EntryOutcome(
            signal_entry=None,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason=reason,
            htf_size_mult=1.0,
            confidence_mult=None,
            ml_conf=None,
        )
    if entry_atr is None or entry_atr == 0:
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="atr_unavail",
            htf_size_mult=1.0,
            confidence_mult=None,
            ml_conf=None,
        )

    # --- HTF 3-state ---
    htf_size_mult = 1.0
    if not sym_cfg.get("mtf_enabled", False) or (sym_cfg.get("mtf_enabled", False) and entry_type == "pullback"):
        htf_decision, htf_size_mult = check_htf_trend(sym_cfg, signal_entry)
        if htf_decision == "block":
            return EntryOutcome(
                signal_entry=signal_entry,
                entry_atr=entry_atr,
                atr=atr,
                entry_type=entry_type,
                mtf_confidence=mtf_confidence,
                gate_open=gate_open,
                regime=regime,
                fused_score=fused_score,
                trend_signal=trend_signal,
                mr_signal=mr_signal,
                blocked=True,
                block_reason="htf_block",
                htf_size_mult=htf_size_mult,
                confidence_mult=None,
                ml_conf=None,
            )
        # soft → allowed at reduced size, not blocked

    # --- Gate 3 ML ---
    ml_passed, confidence_mult, ml_conf = check_ml_gate(sym_cfg, signal_entry, entry_atr)
    if not ml_passed:
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="ml_gate",
            htf_size_mult=htf_size_mult,
            confidence_mult=confidence_mult,
            ml_conf=ml_conf,
        )

    # --- Gate 4 sanity — split so dashboard can attribute volume/spread/tape ---
    if not check_volume_filter(sym_cfg, signal_entry):
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="sanity_volume",
            htf_size_mult=htf_size_mult,
            confidence_mult=confidence_mult,
            ml_conf=ml_conf,
        )
    if not check_spread_filter(sym_cfg):
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="sanity_spread",
            htf_size_mult=htf_size_mult,
            confidence_mult=confidence_mult,
            ml_conf=ml_conf,
        )
    if not check_tape_reading(sym_cfg, signal_entry):
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="sanity_tape",
            htf_size_mult=htf_size_mult,
            confidence_mult=confidence_mult,
            ml_conf=ml_conf,
        )

    # --- MR cooldown ---
    is_mr_entry = mr_signal is not None and not gate_open
    mr_cd_enabled = sym_cfg.get("mr_cooldown_enabled", True)
    mr_cd_window = sym_cfg.get("mr_cooldown_bars", 2) * 3600
    if (
        is_mr_entry
        and signal_entry is not None
        and mr_cd_enabled
        and _st._mr_consecutive_losses.get(symbol, 0) >= 2
        and time.time() - _st._mr_last_loss_time.get(symbol, 0) < mr_cd_window
    ):
        return EntryOutcome(
            signal_entry=signal_entry,
            entry_atr=entry_atr,
            atr=atr,
            entry_type=entry_type,
            mtf_confidence=mtf_confidence,
            gate_open=gate_open,
            regime=regime,
            fused_score=fused_score,
            trend_signal=trend_signal,
            mr_signal=mr_signal,
            blocked=True,
            block_reason="mr_cooldown",
            htf_size_mult=htf_size_mult,
            confidence_mult=confidence_mult,
            ml_conf=ml_conf,
        )

    return EntryOutcome(
        signal_entry=signal_entry,
        entry_atr=entry_atr,
        atr=atr,
        entry_type=entry_type,
        mtf_confidence=mtf_confidence,
        gate_open=gate_open,
        regime=regime,
        fused_score=fused_score,
        trend_signal=trend_signal,
        mr_signal=mr_signal,
        blocked=False,
        block_reason=None,
        htf_size_mult=htf_size_mult,
        confidence_mult=confidence_mult,
        ml_conf=ml_conf,
    )
