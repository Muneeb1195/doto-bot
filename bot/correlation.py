from datetime import datetime

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
from mt5_connect import mt5_call


def fetch_returns_for_symbols(symbols, lookback_hours=24, tf=None):
    datetime.now()
    if tf is None:
        tf = mt5.TIMEFRAME_H1
    rates_dict = {}
    for symbol in symbols:
        rates = mt5_call(mt5.copy_rates_from_pos, symbol, tf, 0, lookback_hours, _timeout=10)
        if rates is None or len(rates) < lookback_hours:
            continue
        df = pd.DataFrame(rates)
        df["ret"] = df["close"].pct_change()
        rates_dict[symbol] = df["ret"]
    return rates_dict


def compute_correlation_matrix(symbols, lookback_hours=24):
    returns_dict = fetch_returns_for_symbols(symbols, lookback_hours)
    if len(returns_dict) < 2:
        return {}
    ret_df = pd.DataFrame(returns_dict).dropna()
    if ret_df.shape[1] < 2 or ret_df.shape[0] < 5:
        return {}
    corr = ret_df.corr(method="pearson")
    result = {}
    for s1 in symbols:
        for s2 in symbols:
            if s1 >= s2 or s1 not in corr.index or s2 not in corr.columns:
                continue
            val = corr.loc[s1, s2]
            result[(s1, s2)] = val
    return result


def get_correlation_reduction(correlation_matrix, new_symbol, existing_symbols, max_reduction=0.5):
    if not correlation_matrix or not existing_symbols:
        return 1.0
    max_corr = 0.0
    for sym in existing_symbols:
        pair = (new_symbol, sym) if new_symbol < sym else (sym, new_symbol)
        val = correlation_matrix.get(pair, 0.0)
        if np.isnan(val) or not np.isfinite(val):
            val = 0.0
        max_corr = max(max_corr, val)
    if np.isnan(max_corr) or max_corr <= 0.5:
        return 1.0
    reduction = 1.0 - ((max_corr - 0.5) / 0.5) * max_reduction
    return max(reduction, 1.0 - max_reduction)
