# ADR 001 — Defer Global Registry Scoping

Date: 2026-08-20
Status: Accepted
Deciders: doto-mt5-bot maintainers

## Context
Candidate 05 in the architecture review (see architecture-plan.md) identified `bot/state.py` as a Singleton/Global Registry with ~33 mutable module globals and a table-driven persistence seam (`PERSISTED`). It blocks parallel tests and scatters cohesion, but it is also load-bearing: 9 modules import it, `save_bot_state` is crash-safety critical, and the live bot is single-threaded with a 10s cycle.

The deeper modules (execution exit tree → `bot/exit_decision.py`, entry gate chain → `bot/entry_policy.py`) already reduce direct coupling by moving decision kernels behind pure interfaces. Further registry scoping would require threading a context object through `main`, `execution`, `filters`, `risk`, and `journal`.

## Decision
Defer full registry scoping (Scoped Contexts / DI container) until after candidates 01–04 land and are soaked live. Keep `state.py` as the process-wide store, but:

- New deep modules receive needed values as arguments and return intents (`ExitIntent`, `EntryOutcome`) rather than mutating globals.
- `state.reset_all()` remains test-only; do not introduce a second global store.
- Persistence stays atomic (`.tmp + fsync + replace`) and table-driven.

## Consequences
- Parallel pytest remains disabled (`state` mutation autouse fixture).
- Future work can migrate one seam at a time (e.g., `ExecutionContext`, `EntryContext`) without a flag-day.
- Architecture reviews should not re-propose a full registry rewrite unless live soak shows concrete pain (lost state on crash, test flake due to leaked globals).

## Alternatives Considered
- Immediate container/DI: rejected — high churn, touches every module, no isolated seam to prove leverage.
- No ADR / ephemeral defer: rejected — would cause re-suggestion next review cycle.

