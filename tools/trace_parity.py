import numpy as np, pandas as pd
from backtest import Backtest

np.random.seed(7)
n = 1500
idx = pd.date_range("2021-01-01", periods=n, freq="h")
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
close = close + np.arange(n) * 0.02
df = pd.DataFrame({
    "time": idx, "open": close - 0.1, "high": close + 0.6,
    "low": close - 0.6, "close": close, "tick_volume": np.full(n, 1000),
}, index=idx)

base = dict(
    symbol="XAU500.raw", initial_balance=400000.0, risk_percent=1.0, commission=0.0,
    ma_type="kama", ema_fast=12, ema_slow=48, atr_period=14, adx_enabled=True,
    atr_sl_mult=2.0, rr=2.0, adx_trend_threshold=25,
    exec_enabled=True, exec_bias_max_flips=3, er_min=0.10, er_period=10,
    mr_enabled=True, mr_rsi_period=14, mr_rsi_oversold=30, mr_rsi_overbought=70,
    htf_ema_slow=200, htf_misalign_size_mult=0.5,
    scoring_enabled=True, scoring_min_entry=0.0,
    correlation_enabled=False, corr_size_mult=1.0,
    dr_enabled=True, ml_enabled=False, volume_filter=False,
)
# volume_filter False (no vol_sma). The parity test leaves volume_filter unset -> default? set explicit
base.setdefault("volume_filter", False)

res = {}
for fast in (False, True):
    bt = Backtest(df=df.copy(), params=dict(base))
    bt.run(fast=fast)
    res[fast] = [(int(t["entry_bar"]), int(t["exit_bar"]), t["exit_reason"], t["type"], round(float(t["pnl"]), 2))
                 for t in bt.trades]
r, f = res[False], res[True]
print("ref trades", len(r), "fast trades", len(f))
k = 0
while k < min(len(r), len(f)):
    if r[k] != f[k]:
        print("FIRST DIVERGENCE at trade index", k)
        print("  REF :", r[k])
        print("  FAST:", f[k])
        for j in range(max(0, k - 3), min(len(r), k + 4)):
            print("   ref[%d]" % j, r[j])
        for j in range(max(0, k - 3), min(len(f), k + 4)):
            print("   fast[%d]" % j, f[j])
        break
    k += 1
else:
    print("identical up to", k, "; ref extra", r[k:], "fast extra", f[k:])
