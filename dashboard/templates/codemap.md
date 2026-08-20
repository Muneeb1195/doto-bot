# dashboard/templates/

## Responsibility
Single-page frontend for the read-only dashboard — no build step. `index.html:1` is a self-contained SPA (inline CSS + inline JS, dark grid layout) that visualizes `data/dashboard_state.json` + `logs/trades.csv` + tail of `logs/bot.log` via the sibling FastAPI `dashboard/api.py`. Rendered by `GET / → HTMLResponse(TEMPLATE_PATH.read_text()):135`.

## Design
- **Zero-tooling static asset** — one 250-line HTML file, `style:7` dark theme (`#0f0f1a/#1a1a2e/#00d4aa`), CSS grid `.metrics` + `table` + `.filter-chart`/`.bar-group/.bar-rows` stacked bars. No `node_modules`, no bundler; safe to `scp` with `scripts/deploy-linux.sh`.
- **Data contract inline** — JS constants `FILTERS:71` (`regime_gate,ml_gate,sanity,no_signal,signals,htf_trend,tail_risk`) with `FILTER_COLOR:72` HSL 40° steps, `rs:73` formats PKR `Rs.x,xx` (matches bot `dashboard.py` PKR display), `EVENT_LABEL:198` maps journal `event`→`Regime` display (SCALE_OUT→Scale-Out, MR_EXIT→Ranging, etc., matching `journal.py` resolved-ticket rule).
- **Polling loop** (`load:75` `Promise.all(fetch /api/state, /api/trades)` + `renderLog:236 fetch /api/logs`, `setInterval 10 s` only when `!document.hidden:245` + `visibilitychange` immediate reload — avoids wasted polls in background tab, mirrors bot 10 s cycle).
- **Renderers** — `renderMetrics:95` balance/equity/floating/margin/positions (+ `renderHealth:105` MT5 connected/server), `renderPositions:112` positions_detail table with profit green/red, `renderFilters:121` filter breakdown table + `renderFilterChart:133` normalized bars `height v/max*100%` with legend/tooltip `title`, `renderPerfMetrics:148` (closed = `event !== OPEN`, WR/PF/avg win-loss/largest/best-worst streak/avg duration 60 m→h), `renderRegimePerf:192` by-event aggregation, `renderTradeLog:212` today-filtered via `toDateString()` + today's PnL summary, `renderRegimeStatus:228` regimes table, `renderLog:236` `<pre.log:35>` max 300 px.
- **API seam** — expects `GET /api/state → load_state() or {}`, `/api/trades → load_trades() sorted exit_time desc`, `/api/logs → {lines:60}` from `dashboard/api.py` (`no-store` on `/api/*`); unauthenticated fetch would 401 (handled by `catch console.error:90`).
- **Responsiveness** — `@media max-width 768` placeholder `36` (layout already `auto-fit minmax 140`) — minimal mobile adaptation.

## Flow
1. Browser `GET /` → `api.index:135` serves raw `index.html`.
2. `load():75` fires on load + every 10 s (if visible) + on visibility return → parallel JSON fetches `/api/state`+`/api/trades` (Basic-auth header injected by browser after initial 401 challenge) → `renderMetrics/Health/Positions/RegimeStatus/Filters/PerfMetrics/RegimePerf/TradeLog` update DOM; `renderLog()` fetches `/api/logs` tail (60 lines of `bot.log` matching `^\d{4}.*\[(INFO|WARNING|ERROR)\]`).
3. Filter bar heights computed `v / max(1, all filter counts)` normalized so cross-symbol comparison is meaningful; today's log derived from `trades.csv` closed rows (append-only journal).

## Integration
- **Served by:** `dashboard/api.py:135` `index()` (`TEMPLATE_PATH = Path(api.py)/templates/index.html:53`, fallback “Template not found”). No direct import of `bot/` — pure file-contract consumer.
- **Consumes:** `/api/state` (produced by `bot/dashboard.write_dashboard_state:atomic .tmp+fsync`, equity 5000), `/api/trades` (derived from `logs/trades.csv` via `journal.py`, Kelly source), `/api/logs` (last 256 KB of `logs/bot.log` via `TimedRotatingFileHandler`).
- **Deployed via:** `scripts/deploy-linux.sh` copies `dashboard/` + `dashboard/templates/` to home-server, `systemd doto-dashboard.service` `uvicorn dashboard.api:app :8501`; health probe `scripts/service-ctl.sh:curl 127.0.0.1:8501 → 401`.
- **Related:** `bot/dashboard.py` (writer), `bot/weekly_summary.py` (same trades source), `.dashboard_public/` static export variant.
