# Archived: local symbol sweepers (2026-08-13)

`sweep_symbols.py` / `sweep_remaining.py` — pre-CI local optimizers that ran
`optimize_params.py --two-phase` per symbol via subprocess and ranked results by
walk-forward score into `logs/symbol_sweep_*.csv`.

## Why archived (superseded)

- **Optimization is CI-only now** (AGENTS.md: "Optimization and ML training run
  ONLY on GitHub Actions"): `optimize.yml` optimizes the `[PORTFOLIO]` symbols
  with the DSR/PBO gate + hybrid failure policy — far beyond the WF-rank CSV
  these produced.
- Both hardcode the **Windows** venv path (`.venv/Scripts/python.exe`); they
  cannot run on the Linux dev box, home-server, or CI runners without edits.
- `sweep_remaining.py` is broken against the current optimizer output (parses
  `score=` as the WF score; the optimizer now prints `WF=`), and hardcodes
  date-stamped filenames plus a stale 23-symbol list including dead symbols
  (IWM, SPY, EURJPY).
- Zero references in code, docs, workflows, or AGENTS.md.

## Reuse path

The orchestration pattern — per-symbol subprocess sweep with WF ranking,
resume-from-CSV (`sweep_remaining.get_done_symbols`), per-symbol crash-safe
progress persistence — is the only copy of that logic in the repo. If symbol
*selection* (ranking candidates beyond the current portfolio) ever moves into
CI, port this pattern into a workflow script rather than restoring these files.
