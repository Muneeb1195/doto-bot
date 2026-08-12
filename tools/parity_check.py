import numpy as np
import pandas as pd
from backtest import Backtest


def bt_df(n=1500, seed=42):
    np.random.seed(seed)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.4)
    highs = closes + np.random.uniform(0.2, 0.8, n)
    lows = closes - np.random.uniform(0.2, 0.8, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": dates,
        "open": closes - np.random.uniform(0.0, 0.3, n),
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": np.random.randint(500, 5000, n),
        "spread": np.random.randint(1, 5, n).astype(float),
    }, index=dates)


PARAMS = dict(
    symbol='XAU500', ma_type='kama', ema_fast=12, ema_slow=48, atr_sl_mult=2.0,
    rr=2.0, adx_trend_threshold=25, risk_percent=1.0, initial_balance=10000,
    htf_misalign_size_mult=0.5, scoring_enabled=True, scoring_min_entry=0.60,
    volume_filter=True, volatility_filter=True, spf_enabled=True, dr_enabled=True,
    dr_vol_adjust=True, ml_enabled=True, tr_enabled=True, mr_enabled=True,
    pb_enabled=True, chandelier_enabled=True, scale_out_enabled=True,
    session_enabled=False, atr_period=14, adx_enabled=True,
)


def main():
    df = bt_df()
    res = {}
    for fast in (False, True):
        bt = Backtest(df=df.copy(), params=dict(PARAMS))
        bt.run(fast=fast)
        eq = np.array(getattr(bt, 'equity', []), dtype=float)
        tr = bt.trades
        pnls = [round(float(t['pnl']), 4) for t in tr]
        res[fast] = (len(tr), round(float(eq[-1]), 4), pnls)
        print('fast=', fast, 'trades=', res[fast][0], 'eq_final=', res[fast][1])
    print('PNL MATCH:', res[False][2] == res[True][2])
    print('EQ MATCH:', res[False][1] == res[True][1])


if __name__ == "__main__":
    main()
