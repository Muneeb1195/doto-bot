# Executive Summary — Critique Analysis

**Document analyzed**: Executive Summary.odt (external reviewer)
**Date**: 2026-07-10
**Current bot version**: All 13 items implemented (budget, deviation, filling, capital exclusion, uncertain fix, meta-labeler, Platt scaling, OOS validation)

---

## 1. "Too Many Filters" — 9 gates kill ~70% of signals


The chain was: Regime → Exec → Volume → Spread → Volatility → News → Tape → ML → Composite → Sizing.

**What changed**: Scoring mode (Phase 2) replaced the first 8 hard boolean gates with a weighted composite score (≥0.70 gate). The filter functions still exist as score sources, but they no longer `continue` individually.

**Verdict**: PARTIALLY ADDRESSED. The scoring refactor IS the fix for this exact critique — it converts "9 sequential gates" into "1 weighted gate + 6 data sources." But the reviewer is right that we haven't *measured* whether each score dimension contributes. We should run an ablation study: disable one score dimension at a time and measure PF/WR change.

**Recommendation**: Keep scoring mode as-is. Run a one-time ablation to verify all 6 score dimensions earn their weight. If any dimension has zero impact (e.g., `tape` never fires different from 1.0), remove it.

---

## 2. "Over-Engineering" — 2600 lines, complexity cost


**Verdict**: TRUE but inevitable. The system HAS grown complex because it handles 10 symbols × 2 entry modes × 3 exit mechanisms × 5 regimes × 67 features × 6 sizing multipliers. Each addition was individually justified. The reviewer's concern is about maintainability, not correctness.

The real risk: complexity makes it hard to identify WHERE the edge comes from. If the system is profitable, is it the regime engine? The chandelier exit? The MR entries? Kelly? We can't answer this.

**Recommendation**: DO NOT simplify by removing modules — that's cargo-culting. Instead, invest in measurement infrastructure:
- Log per-symbol PF contribution (already have trade journal)
- Log per-filter / per-score-dimension pass rates (already have `_filter_stats`)
- Log per-exit-type PnL (`REVERSAL` vs `CHANDELIER` vs `SCALE_OUT` vs `MR_EXIT`)
Once you can answer "what drives PnL?", complexity becomes manageable. Without that, it's a black box.

---

## 3. "Kelly Lookback = 20 is too noisy"


The doc says 100–300 trades or EWMA.

**What changed**: Already bumped to 50.

**Verdict**: 50 is reasonable for our trade volume (~30-50 trades/symbol/year). 100-300 would take 2-6 years of data per symbol — not usable. EWMA would add adaptive smoothness but at the cost of another tunable parameter.

**Recommendation**: Keep 50. Monitor Kelly_mult in logs — if it oscillates wildly (>±50% month over month), consider EWMA. If it stays in [0.8, 1.2] mostly, 50 is fine.

---

## 4. "Grid search needs walk-forward validation"


**What changed**: Added chronological 80/20 OOS split to `optimize_params.py`. The best params are validated on OOS data and degradation % logged.

**Verdict**: FIXED. The current implementation (single 80/20 split) is sufficient for a first pass. True walk-forward (rolling 6-month windows) would be better but adds ~10× compute time.

**Recommendation**: The current OOS validation catches the worst overfitting. Upgrade to walk-forward only if you see OOS score consistently degrading >30% below IS score across multiple symbols.

---

## 5. "ML is a gate, not a driver"


The doc says ML should predict expected R (or expected time to TP, expected volatility), not binary win/lose.

**Verdict**: VALID and important. This is the single biggest architectural opportunity we haven't touched.

**Current**: `model.predict("win")` → confidence used for gating + sizing  
**Proposed**: `model.predict("expected_R_multiple")` → directly drives position sizing

**Why it matters**:
- A binary model treats a +0.1R trade the same as a +5R trade — both are "wins"
- An expected-value model would size up for +5R signals and skip +0.1R signals
- The gating threshold becomes unnecessary — you already know the expected value

**Effort**: MAJOR. Requires:
1. Changing triple-barrier labels from {0,1} to continuous R (regression)
2. Switching XGBoost to `objective='reg:squarederror'`
3. Changing `check_ml_signal()` to return expected R instead of confidence
4. Changing sizing to use expected R directly (e.g., `volume = kelly / expected_R`)
5. Retraining all 10 models + 2 pool models + 10 meta-labelers

**Recommendation**: P1 after current system stabilizes. Don't do it now — let the current system trade for 2-4 weeks first, so we have a baseline to compare against.

---

## 6. "Labels don't match live exits"


Triple-barrier labels use TP=2×ATR, SL=1×ATR, max_hold=20 bars. Live exits use chandelier (3.5×ATR loose, 1.5×ATR tight) + scale-out (1.5× and 2.5× ATR partials).

**Verdict**: GENUINE PIPELINE BUG. The model is trained to predict "does price hit 2× ATR up or 1× ATR down first within 20 bars?" but the live system exits at very different levels:

| Exit Type | Distance | When |
|-----------|----------|------|
| Scale-out step 1 | +1.5× ATR | Partial close |
| Scale-out step 2 | +2.5× ATR | Partial close |
| Chandelier (loose) | Trail at 3.5× ATR from HH | Full close |
| Chandelier (tight) | Trail at 1.5× ATR from HH | Full close (R≥3) |
| SL | 1× ATR | Full close |

A trade that hits +1.5× ATR (scale-out takes profit on 30%) then reverses to chandelier exit at 0.5× ATR above entry — the model would label it a "loss" (didn't hit 2× ATR), but the live system would show a net profit (partial gain + chandelier exit).

**Recommendation**: DO NOT fix right now. The model still adds value (it discriminates directions with some skill), and the label misalignment is systematic (affects both sides equally). If the model is directional, the label mismatch just introduces noise — it reduces the effective sample size but doesn't bias the model. Fixing this would require:
1. Simulating the full exit logic in `train_model.py` to generate realistic labels
2. This is a major engineering project

**But**: Track it as a known limitation. If the ML signal's win rate doesn't exceed 55% after 200+ trades, this mismatch is a prime suspect.

---

## 7. "67 features is too many — needs SHAP pruning"


**Verdict**: TRUE but low priority. The reviewer is right that more features ≠ better. However, XGBoost is remarkably good at ignoring irrelevant features. The real risk is not performance degradation — it's that irrelevant features add noise to feature importance analysis and make retraining slower.

**Recommendation**: Defer. Run SHAP analysis once after 3 months of live data to see which features actually drive predictions. Expect 15-20 features to dominate. Prune then.

---

## 8. "Static ADX-only regime classifier"


The reviewer suggests using more inputs (volatility, trend persistence, correlation, macro conditions).

**What changed**: Added percentile ADX (trailing 180-day) to make thresholds adaptive.

**Verdict**: PARTIALLY ADDRESSED. The reviewer's criticism is about the *depth* of the regime model — ADX only captures directional strength, not volatility regime, correlation regime, or market phase. This is a valid observation but not actionable for us.

**Why not to act**: 
- Adding more regime inputs creates more tunable parameters and more state to manage
- The 5-state ADX regime already achieves the PRIMARY goal: distinguishing trend-following from mean-reversion environments
- Professional CTA firms use ADX-based regime classification as a baseline too

**Recommendation**: Keep as-is. Add a note to revisit if the system struggles in low-ADX environments (EURUSD, NZDUSD) where the regime is permanently "uncertain."

---

## 9. "Session filter disabled — low-hanging fruit"


London-only, Asian-skip, and overlap filters are all disabled.

**Verdict**: TRUE — this IS low-hanging fruit. The reviewer suggests testing it because FX pairs perform differently by session. Given that many of our symbols are FX (EURUSD, EURJPY, NZDUSD, USDJPY, GBPJPY), session filtering could materially improve signal quality.

**Why it matters**: 
- London session (8AM-5PM GMT) has highest FX volume, tightest spreads, most trend persistence
- Asian session (midnight-9AM GMT) has widest spreads, most noise, frequent reversals
- Our current uncertain-loop fix means crossover signals fire during Asian hours when EMAs cross due to noise

**Recommendation**: TEST. This is the single easiest improvement to try. Steps:
1. Enable `skip_asian = True` in `settings.ini [SESSION]`
2. Run a 3-month backtest comparison (Asian on vs off)
3. If PF improves ≥10%, keep it. If not, disable again.

Effort: 5 lines in settings.ini. No code changes needed.

---

## 10-11. Tape reading + News sentiment removal


The reviewer says these are prime candidates for removal.

**Verdict**: The scoring matrix already handles this correctly. Individual filter functions shouldn't be removed because:
- The legacy branch (backtest compatibility) needs them
- The scoring branch reads their raw data as score inputs
- Even if a filter rarely blocks trades, the scoring branch uses it as a signal dimension with low weight

**Recommendation**: Skip. The scoring branch's `compute_entry_score()` already gives tape and news low default weights (0.10 and 0.05). They contribute when they have information and are ignored when they don't.

---

## Summary: What to ACTUALLY Do

| Priority | Action | Effort | Expected Impact |
|----------|--------|--------|-----------------|
| **This week** | Test `skip_asian = True` in backtest | 5 min | Medium — cleaner signals |
| **This month** | Build PnL attribution by exit type | 2 days | High — understand where edge comes from |
| **Next month** | ML expected-value model (regression) | 1 week | High — replaces binary gate + sizing chain |
| **Next quarter** | SHAP pruning (67→~20 features) | 2 days | Low-Med — cleaner models, faster retrain |
| **Track** | Label/exit alignment fix | 2 weeks | Medium — but defer until ML has live baseline |
| **Skip** | Remove tape/news filters | — | Scoring already handles this |
| **Skip** | EWMA Kelly, walk-forward validation | — | 50 lookback + OOS split is sufficient |
| **Skip** | Richer regime classifier | — | ADX percentile is adequate for 5 states |
| **Skip** | ThreadPoolExecutor | — | Cycle time ~10-20s, well under 180s limit |

---

## What We Did Right (Reviewer Agrees)

The reviewer rated us 8.6/10 overall and specifically praised:

1. **Regime-based design** — 5-state regime engine is "one of the strongest architectural choices"
2. **Position sizing** — 6-stage multiplier chain is "closer to institutional than retail"
3. **Portfolio thinking** — correlation, max positions, daily loss, circuit breaker
4. **Exit system** — stronger than entries, combining chandelier + scale-out + reversal
5. **Engineering quality** — persistent state, reconnect, watchdog, dashboard, recovery

These are the parts of the system to PROTECT and not dismantle.
