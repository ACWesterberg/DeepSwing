# DeepSwing — Implementation Status

Last updated: 2026-07-01

---

## Done ✅

### Phase 1 — Foundation
- [x] Project scaffolding, directory structure, `__init__.py` files
- [x] `requirements.txt` (Python 3.11, all Pi-safe dependencies)
- [x] `.env.example` with all required API key slots + model/backup overrides
- [x] `config/settings.py` — Pydantic Settings, dual-track config, risk params, watchlists, model IDs, MIPRO backup + preflight toggles
- [x] `src/db.py` — SQLAlchemy models: Trade, Position, PortfolioSnapshot, Heuristic, Decision (all with `track` column)
- [x] `src/analysis/technical.py` — 11 indicators via `ta` library: EMA/SMA, ATR, Bollinger Bands, RSI, Parabolic SAR, EOM, OBV, Fibonacci
- [x] `src/analysis/regime.py` — Hurst Exponent (R/S analysis) + lag-1 autocorrelation; trending/mean-reverting/neutral classification
- [x] Database init (`init_db()`)

### Phase 2 — Core Agent
- [x] `src/analysis/screener.py` — multi-factor filter (SMA, RSI, volume, regime); weighted scoring; top-N candidates
- [x] `src/agent/risk.py` — ATR-based stop validation, RRR check, 1% position sizing, drawdown-mode halving, duplicate-ticker check, per-sector position cap
- [x] `src/agent/memory.py` — file-backed heuristic store; track-namespaced; retrieve by regime/market relevance; prune; promote core rules
- [x] `src/agent/decision.py` — DSPy `TradeDecision` (BUY/PASS) + `ExitDecision` (HOLD/SELL) signatures; `DecisionEngine` per track; loads compiled program if available; `dspy.context()` per call; `build_lm()` applies reasoning-model params
- [x] `src/agent/news_analyzer.py` — keyword pre-filter → shared GPT news analysis (Swedish + English)

### Phase 3 — Simulation + ERL + DSPy Optimization
- [x] `src/portfolio/simulator.py` — track-tagged paper portfolio; open/close with slippage; trailing stop; stop-loss/take-profit auto-close; drawdown-mode flag; `entry_inputs` captured on positions/trades
- [x] `src/portfolio/metrics.py` — Sharpe, max drawdown, win rate, avg RRR, total return, `optimization_metric = win_rate × avg_rrr`
- [x] `src/agent/erl.py` — post-trade causal analysis; Claude Opus + extended thinking (Claude); GPT-5.5 + `reasoning_effort` (GPT); structured heuristic extraction + storage
- [x] `src/scheduler/optimizer.py` — weekly MIPROv2 per track; P&L-weighted metric; split prompt-model (heavy proposer) / task-model (decision tier); archives previous compiled program; `DecisionEngine.reload()`; offsite backup; heuristic prune/promote

### Phase 4 — Scheduler + Data Ingestion
- [x] `src/scheduler/market_hours.py` — `is_market_open()` (scan window), `is_exchange_open()` (badge, true exchange hours), `active_markets()`, CET-aware
- [x] `src/scheduler/scan_loop.py` — full scan cycle; VIX circuit-breaker; per-position-market FX conversion to SEK; capacity-aware scanning (skips the candidate/news/decision pipeline for tracks with no free cash, drops to a holdings-only monitor when all tracks are fully allocated); jump-triggered news exits (news + AI exit review only fire once a holding moves ≥ `holdings_news_jump_pct`); WebSocket trade events; decision persistence
- [x] `src/data/` — now thin wrappers over the shared **`financedata`** package: `market_data`, `news_fetcher`, `insider_fetcher`, `macro_data`; `universe.py` + `config/universe.csv` drive the Nordic watchlist (OMXS/OSLO/OMXH/OMXC)
- [x] FX / currency handling — `_to_sek_price` + suffix→currency map (.ST/SEK, .OL/NOK, .HE/EUR, .CO/DKK, US/USD); per-position-market conversion

### Phase 5 — Dashboard
- [x] `src/dashboard/app.py` — FastAPI; REST: `/api/status`, `/portfolio`, `/trades`, `/comparison`, `/heuristics`, `/decisions`, `/decisions/history`, `/prompts`, `POST /scan`, `POST /reset`, `POST /backtest`; WebSocket `/ws`; cookie-session auth
- [x] `src/dashboard/templates/index.html` — tabs: Comparison, Claude, GPT, Decisions, Heuristics (both), Prompts
- [x] `src/dashboard/static/` — Chart.js equity overlay, head-to-head table, positions/trades, heuristic cards, decision feed + history, scan buttons + progress toast, auto-refresh + WebSocket push
- [x] `main.py` — DB init, boot preflight (log model config + ping models), APScheduler (15-min scan + Sunday 02:00 MIPRO), uvicorn
- [x] `systemd/deepswing.service` — autostart on Pi boot, Pi 5 resource limits

### Reliability & Ops (this cycle)
- [x] **ERL / MIPRO input capture** — trade-entry DSPy inputs captured in `decision.py`, stored on `OpenPosition.entry_inputs`, carried to `ClosedTrade`, consumed by `optimizer.py` (previously the trainset was always empty)
- [x] **P&L-weighted MIPRO metric** — `_pnl_weighted_metric` scores decisions by realized return, not binary action-match
- [x] **MIPRO offsite backup** — `src/scheduler/backup.py` commits/pushes each compiled program (history + `latest.json` + metrics) to a standalone git repo
- [x] **Boot preflight** — `src/scheduler/preflight.py` logs resolved model IDs and pings each model once so bad IDs/creds surface at startup
- [x] **Model upgrades** — scan: Sonnet 5 / GPT-5; ERL: Opus 4.8+thinking / GPT-5.5+reasoning; news: GPT-5-mini (shared); MIPRO proposer: Opus 4.8 / GPT-5.5; `build_lm` fixes reasoning-model params
- [x] **ERL environment context** — entry-time news + macro now passed into ERL so heuristics can attribute outcomes to the market environment
- [x] **Market-wide news environment** — `fetch_market_headlines` pulls the full RSS feed (not ticker-filtered) once per scan; folded into `macro_context`, so geopolitics/sector/risk themes reach decisions, ERL, and MIPRO
- [x] **Earnings-proximity filter** — candidates within `earnings_buffer_days` (default 2) of earnings are dropped before decisions (financedata fundamentals + `ts_to_days`)
- [x] **Bug fixes** — cross-market FX contamination; Nordic currency mis-mapping; market-status badge (exchange hours vs scan window); DSPy thread error (`dspy.context()`); GPT-5 `dspy.LM` crash
- [x] **Tests** — technical, regime, screener, risk, scan_loop (integration), e2e lifecycle, backtesting, backup, optimizer, preflight, decision_lm, watchlist, insider, reset (196 passing)

### Documentation & Deployment
- [x] `SETUP.md`, `README.md`, `ARCHITECTURE.md`, `STATUS.md`, `CLAUDE.md`
- [x] `.gitignore` — excludes `.env`, `venv/`, `data/*.db`, `heuristics/`, `compiled/`
- [x] Deployed and running on Pi 5; Cloudflare Tunnel live (`trade.westerberg.dev`); dashboard cookie auth

---

## To Do 🔲

### Improvements
- [ ] **Sector correlation matrix** — a per-sector position *count* cap is enforced; the true 0.7 max-correlation rule (yfinance sector tags → correlation matrix) is not yet implemented
- [ ] **`_fix_rrr` masks target discipline** — auto-stretching targets in the 1.0–2.0 RRR band means MIPRO never learns to place good targets, only to avoid broken stops; consider learning target placement instead
- [ ] **News model on reasoning tier** — `gpt-5-mini` may spend budget on reasoning; monitor Swedish news summary quality, bump model or tune `max_completion_tokens` if weak

### Pi Deployment / Ops
- [ ] Verify APScheduler fires correctly across DST changes (Stockholm CET↔CEST)
- [ ] Monitor memory usage during the first weekly MIPRO run (Pi 5, 1G cap)
- [ ] Complete the MIPRO backup repo setup on the Pi (`MIPRO_BACKUP_REPO_DIR`) before the first MIPRO run
- [ ] Fix the committed `systemd/deepswing.service` paths (still `/home/pi/DeepSwing`; actual Pi path is `/home/alexander/Documents/DeepSwing`)

### After First 30+ Closed Trades
- [ ] Verify the first MIPRO run produces a valid compiled JSON (and that the backup fires)
- [ ] Compare `optimization_metric` (win_rate × avg_rrr) pre- vs post-MIPRO
- [ ] Review ERL heuristics for quality — specific and actionable?
- [ ] Track Claude vs GPT divergence on the same candidates

---

## Known Limitations

| Item | Detail |
|---|---|
| MIPRO sample size | `auto="light"` on ~30 trades (24 train / 6 val) yields calibration, not transformation; expect modest gains until trade count grows |
| Portfolio state is in-memory | `closed_trades`/positions live in the running process and are not rehydrated from the DB on restart; a restart resets MIPRO's available trainset |
| Reasoning-model IDs | GPT-5/5.5 and Claude 5 IDs are env-overridable; a wrong ID surfaces at boot via preflight but still requires a manual `.env` fix |
| Non-SEK Nordic FX | Depends on `financedata`'s `to_sek`/`get_fx_rate`; if an FX rate is unavailable the raw native-currency price is used (logged) |
