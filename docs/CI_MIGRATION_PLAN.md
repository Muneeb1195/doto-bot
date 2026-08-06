# CI Migration Plan — Offload Training + Optimization to GitHub Actions

## Goal
Move ML model training and weekly/monthly optimization from NUC to GitHub Actions.
NUC runs only live trading. Repo goes public (4 vCPU/16GB free).
Git LFS for M1 data (~700 MB, within 1 GB free tier).
Discord webhook for CI notifications.

## Execution Order

### Phase A — LFS + Repo Setup
1. `git lfs install` (one-time per machine)
2. Create `.gitattributes`: `data/history/*.csv filter=lfs diff=lfs merge=lfs -text`
3. Remove `data/history/` from `.gitignore`
4. `git add data/history/` + commit ("chore: track MT5 history CSVs via Git LFS")
5. `gh secret set DISCORD_WEBHOOK --body "https://discord.com/api/webhooks/1523693018018873424/FAyVjZx3OAi3ArKtIynlmtihPzeU1qP3LynH5vvmakezCSRY2oeVhyWlTUK2261ecN7B" --repo Muneeb1195/doto-bot`
6. `gh repo edit Muneeb1195/doto-bot --visibility public`

### Phase B — Scripts (NUC side)
7. Create `scripts/export_mt5_data.py` — MT5 → CSV export (H1/M15/M1)
8. Create `scripts/download_models.py` — release → models/params → restart bot

### Phase C — Bot Code Changes
9. Add `--csv` flag to `bot/train_model.py`
10. Add `--csv` flag to `bot/optimize_params.py`

### Phase D — CI Workflows
11. Create `.github/workflows/train.yml`
12. Create `.github/workflows/optimize.yml`
13. Modify `.github/workflows/ci.yml` (add cache for data/history)

### Phase E — Verify + Cleanup
14. Commit + push
15. Verify CI test still green
16. Create NUC timers (export-mt5-data.timer, download-models.timer)
17. Manually trigger train + optimize workflows
18. Verify releases created, NUC downloads, bot restarts
19. Disable old DotoOptimizer/DotoRetrain tasks on NUC

## File Specs

### scripts/export_mt5_data.py
- Export H1 + M15 + M1 bars from MT5 terminal → data/history/<symbol>_<tf>.csv
- Reference: bot/parallel_optimize.py:_fetch_csv_mode() for MT5→CSV logic
- CSV columns: time,open,high,low,close,tick_volume,spread (epoch seconds for time)
- Portfolio symbols: BTCUSD.raw, US30.raw, GBPJPY.raw, SOLUSD.raw, XRPUSD.raw, EURUSD.raw, US500.raw, XAUUSD.raw
- For each symbol, export 3 files: <symbol>_H1.csv, <symbol>_M15.csv, <symbol>_M1.csv
- Graceful no-op if MT5 not connected (don't crash the timer)
- Use mt5.copy_rates_range() for H1/M15, mt5_connect.fetch_rates_paged() for M1
- Save point/tick_value/vstep to settings.ini [SYMBOL_POINTS] for offline reconstruction

### scripts/download_models.py
- Check `gh release list` for latest tag vs local marker `.last_release_tag`
- If newer:
  - Download `models.tar.gz` → extract to `models/`
  - Download `strategy-params.json` → for each symbol, call `update_symbol_strategy()` from bot.auto_optimizer
  - `systemctl --user restart doto-bot`
  - Update `.last_release_tag`
- Exit cleanly if no new release or no gh auth

### bot/train_model.py (--csv flag)
- When `--csv`: use `load_csv_data()` from parallel_optimize instead of `mt5.copy_rates_range()`
- Load M15 from `data/history/<symbol>_M15.csv` for MTF features
- Load M1 from `data/history/<symbol>_M1.csv` for orderflow feature backfill
- Rest of training pipeline unchanged

### bot/optimize_params.py (--csv flag)
- Add `--csv` flag, thread through `fetch_data()` → `load_csv_data()`
- When `--csv`: no MT5 connection needed

### .github/workflows/train.yml
```yaml
name: Train Models
on:
  schedule:
    - cron: '0 3 * * 0'  # Sun 03:00 UTC
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - run: python bot/train_model.py --csv --symbols ALL --years 2
      - name: Package models
        run: |
          tar czf models.tar.gz models/*.pkl models/*.calib.npz
      - name: Create release
        run: |
          TAG="train-$(date +%Y%m%d-%H%M)"
          gh release create "$TAG" models.tar.gz \
            --title "ML Models $(date +%Y-%m-%d)" \
            --notes "Weekly retrain"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Notify Discord
        if: always()
        run: |
          STATUS="${{ job.status }}"
          curl -H "Content-Type: application/json" \
            -d "{\"content\":\"ML Training: $STATUS\"}" \
            ${{ secrets.DISCORD_WEBHOOK }}
```

### .github/workflows/optimize.yml
```yaml
name: Optimize Strategy
on:
  schedule:
    - cron: '0 2 * * 0'       # Weekly Sun 02:00 UTC
    - cron: '0 2 1-7 * 0'     # Monthly CPCV (1st Sun)
  workflow_dispatch:

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Determine mode
        id: mode
        run: |
          DOM=$(date +%u)  # day of week
          DOW=$(date +%d)  # day of month
          if [ "$DOW" -le 7 ] && [ "$DOM" -eq 7 ]; then
            echo "mode=monthly" >> $GITHUB_OUTPUT
          else
            echo "mode=weekly" >> $GITHUB_OUTPUT
          fi
      - name: Run optimizer
        run: |
          python bot/auto_optimizer.py --csv --symbols ALL --years 2 \
            --mode ${{ steps.mode.outputs.mode }}
      - name: Package params
        run: |
          # Extract best params from CSV → strategy-params.json
          python -c "
          import csv, json
          # Read optimize CSVs, extract best row per symbol
          # Output strategy-params.json with symbol→params mapping
          "
      - name: Create release
        run: |
          TAG="optimize-$(date +%Y%m%d-%H%M)"
          gh release create "$TAG" strategy-params.json \
            --title "Strategy Params $(date +%Y-%m-%d)" \
            --notes "Weekly optimization"
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - name: Notify Discord
        if: always()
        run: |
          curl -H "Content-Type: application/json" \
            -d "{\"content\":\"Optimization: ${{ job.status }}\"}" \
            ${{ secrets.DISCORD_WEBHOOK }}
```

### .github/workflows/ci.yml (MODIFY)
- Keep lint, typecheck, shellcheck, secret-scan, test
- Add to lint job:
```yaml
- uses: actions/cache@v4
  with:
    path: data/history
    key: mt5-history-${{ hashFiles('data/history/*.gitattributes') }}
```
- Keep workflow_dispatch: trigger (needed since push-trigger unreliable)

### .gitignore (MODIFY)
- Remove the `data/history/` line

### .gitattributes (NEW)
```
data/history/*.csv filter=lfs diff=lfs merge=lfs -text
```

## Key Reused Functions

- `update_symbol_strategy(symbol, rec, settings)` — bot/auto_optimizer.py:114
  - Maps CSV keys → settings.ini: ema_fast→ema_fast_period, sl→atr_sl_multiplier, etc.
  - Uses `write_settings()` for atomic write
- `load_csv_data(symbol)` — bot/parallel_optimize.py:124
  - CSV → DataFrame parser
- `_fetch_csv_mode(args)` — bot/parallel_optimize.py:542
  - Reference for MT5→CSV export logic

## Param Mapping (existing)
| CSV key | settings.ini key | type |
|---------|-----------------|------|
| ema_fast | ema_fast_period | int |
| ema_slow | ema_slow_period | int |
| sl | atr_sl_multiplier | float |
| rr | risk_reward_ratio | float |
| adx | adx_trend_threshold | int |
| score | scoring_min_entry | float |

## Verification Checklist
- [ ] `ruff check bot/ scripts/` passes
- [ ] `mypy bot/ services/` passes
- [ ] `git lfs ls-files` shows CSV files
- [ ] CI test green
- [ ] Train workflow completes + creates release
- [ ] Optimize workflow completes + creates release
- [ ] NUC download script picks up release + restarts bot
- [ ] Discord receives notification

## NUC Timers (systemd user units)

### export-mt5-data.timer
```
[Unit]
Description=Export MT5 data to CSV for CI training

[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### export-mt5-data.service
```
[Unit]
Description=Export MT5 bars to CSV and push to git

[Service]
Type=oneshot
WorkingDirectory=/home/muneeb/doto-mt5-bot
ExecStart=/home/muneeb/doto-mt5-bot/.venv/bin/python scripts/export_mt5_data.py
```

### download-models.timer
```
[Unit]
Description=Check for new trained models/params releases

[Timer]
OnCalendar=*-*-* *:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

### download-models.service
```
[Unit]
Description=Download latest models/params from GitHub releases

[Service]
Type=oneshot
WorkingDirectory=/home/muneeb/doto-mt5-bot
ExecStart=/home/muneeb/doto-mt5-bot/.venv/bin/python scripts/download_models.py
```
