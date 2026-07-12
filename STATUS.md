# DeepSwing — Implementation Status

Last updated: 2026-07-12

---

## Done ✅

### Phase 6 — Options tracks (claude-opt / gpt-opt)
- [x] `src/analysis/options_math.py` — closed-form Black-Scholes price/delta/theta (`math.erf`, no scipy)
- [x] `src/data/options_chain.py` — yfinance US chain fetch; DTE window + delta band + OI/volume/spread liquidity gates → ≤8-contract shortlist; quote refresh per (underlying, expiry); prompt formatting
- [x] `src/agent/options_decision.py` — DSPy `OptionTradeDecision` (BUY/PASS + contract index + premium-relative exit plan); per-track engine; loads `compiled/{track}_option_decision.json`
- [x] `src/agent/options_risk.py` — premium-budget sizing (1% of equity, 2% single-contract hard cap), reward/risk ≥ 2.0, duplicate-underlying block, liquidity re-check, drawdown halving
- [x] `src/portfolio/options_simulator.py` — `OptionsPortfolio` (long single-leg calls, ×100 multiplier, SEK premiums); profit-target/premium-stop/time-stop sweep; expiry detection; durable state (same `portfolio_state` table)
- [x] `src/scheduler/options_scan.py` — hourly US-session scan sharing the stock scan lock; fill at mid + adverse half-spread; daily 22:10 CET expiry sweep settling at intrinsic; options-flavored ERL trigger
- [x] `src/scheduler/optimizer.py` — `run_options_mipro` with option-scaled P&L metric (k=2, premium-relative returns); weekly slot shared with stock tracks
- [x] Dashboard — 4-track comparison chart + head-to-head table, options track tabs, options prompts panels, `POST /api/scan/options`, reset covers all tracks
- [x] `tests/test_options.py` — Black-Scholes, chain filters, risk sizing, simulator lifecycle (open/close/sweep/expiry/state roundtrip)

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
- [x] `src/portfolio/persistence.py` — durable portfolio state: full live state (cash, open positions, closed trades, peak equity, next trade id) mirrored to the `portfolio_state` table on every open/close + end of scan, rehydrated on startup so tracks survive a redeploy; `/api/reset` clears persisted rows
- [x] `src/agent/erl.py` — post-trade causal analysis; Claude Opus + extended thinking (Claude); GPT-5.5 + `reasoning_effort` (GPT); structured heuristic extraction + storage
- [x] `src/scheduler/optimizer.py` — weekly MIPROv2 per track; P&L-weighted metric; split prompt-model (heavy proposer) / task-model (decision tier); archives previous compiled program; `DecisionEngine.reload()`; offsite backup; heuristic prune/promote

### Phase 4 — Scheduler + Data Ingestion
- [x] `src/scheduler/market_hours.py` — `is_market_open()` (scan window), `is_exchange_open()` (badge, true exchange hours), `active_markets()`, CET-aware
- [x] `src/scheduler/scan_loop.py` — full scan cycle; VIX circuit-breaker; per-position-market FX conversion to SEK; capacity-aware scanning (skips the candidate/news/decision pipeline for tracks with no free cash, drops to a holdings-only monitor when all tracks are fully allocated); jump-triggered news exits (news + AI exit review only fire once a holding moves ≥ `holdings_news_jump_pct`); non-blocking manual scans (`/api/scan` offloaded via `run_in_executor`) serialized by a `_scan_lock` so manual + scheduled can't overlap/double-open; WebSocket trade events; decision persistence
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
- [x] **Durable portfolio state** — live portfolios mirrored to `portfolio_state` and restored on startup, so tracks survive a redeploy (previously reset to starting capital on every `systemctl restart`)
- [x] **Non-blocking scans** — `/api/scan` runs `run_scan` in a worker thread so a scan no longer freezes the dashboard event loop; `_scan_lock` serializes scans so manual + scheduled can't overlap
- [x] **NewsAPI resilience** — per-ticker cache + a 429 breaker (skip NewsAPI → RSS for a cooldown), plus a free per-ticker fallback (yfinance/Yahoo, Finnhub-preferred for US when keyed) so US tickers still get news
- [x] **Volume screened on the completed daily bar** — fixes the screener passing 0 candidates every morning (partial forming bar read ~0.1× and failed the `volume_spike_multiplier` gate)
- [x] **Universe hygiene** — disabled 3 delisted Nordic tickers (TFBANK.ST, SKAKO.CO, ILKKA2.HE) that logged a yfinance ERROR on every scan
- [x] **Tests** — technical, regime, screener, risk, scan_loop (integration), e2e lifecycle, backtesting, backup, optimizer, preflight, decision_lm, watchlist, insider, reset (196 passing). Note: this cycle's ops features (persistence, scan lock, news breaker/fallback, volume fix) are verified manually but not yet in the suite.

### Documentation & Deployment
- [x] `SETUP.md`, `README.md`, `ARCHITECTURE.md`, `STATUS.md`, `CLAUDE.md`
- [x] `.gitignore` — excludes `.env`, `venv/`, `data/*.db`, `heuristics/`, `compiled/`
- [x] Deployed and running on Pi 5; Cloudflare Tunnel live (`trade.westerberg.dev`); dashboard cookie auth

### Correctness & security review fixes (2026-07-02)
- [x] **VIX halt no longer abandons holdings** — a VIX ≥ 35 halt blocks new entries but falls through to the holdings monitor, so stops/targets/news exits still run during volatility spikes
- [x] **ATR-scaled trailing stop + correct exit labels** — the fixed 2% trail (tighter than most tickers' daily ATR; killed winners long before the RRR 2.0 target) is now `trailing_stop_atr_multiplier` (2×ATR, SEK-converted at entry, persisted per position); trailed exits close as `exit_reason="trailing_stop"` instead of being mislabeled `"stop_loss"`, so ERL no longer analyzes profitable trailed exits as stop-outs
- [x] **ATR stop-sanity check fixed** — `stop < atr_stop * 0.90` applied 10% of *price* as slack (toothless) and mixed SEK entry prices with native-currency ATR; now compares stop distance vs 1.5×ATR as fractions of price (currency-safe, 10% slack on the ATR distance)
- [x] **Position-value cap** — risk-based sizing is unbounded with tight stops (position could exceed cash and the approved BUY silently vanished at execution); position value is now capped at `max_position_pct` (25%) of equity and at available cash; execution-time failures land in the decisions feed as BLOCKED
- [x] **US market hours in ET** — the fixed 15:30–22:00 CET window missed the first NYSE hour (or overshot the close) during the ~3 weeks/year when US and EU DST are out of sync; US windows are now evaluated in America/New_York
- [x] **FX guard** — `_to_sek_price` returns `None` when conversion is unavailable instead of silently booking raw USD/EUR prices against the SEK book; entries are BLOCKED, price updates skipped
- [x] **ERL off the scan thread** — ERL (extended-thinking call, potentially minutes per closed trade) ran inline in the scan despite "non-blocking" claims; now runs in daemon threads (`wait_for_erl()` for tests/shutdown)
- [x] **Sharpe honesty** — per-trade returns were annualized as if daily (×√252, overstating several-fold); now scaled by the actual average holding period; `/api/comparison` equity curves get a live mark-to-market point so open P&L is visible in the head-to-head chart
- [x] **Heuristic count calibration** — access counts increment at most once per hour per heuristic (were inflated by every 15-min scan × candidate, entrenching early rules); prune gets a 7-day grace period so new rules aren't culled before they can be used
- [x] **Dashboard security** — session cookie was the plaintext password (irrevocable if leaked); now a random server-side token. WebSocket `/ws` bypassed the auth middleware entirely (BaseHTTPMiddleware only sees http scope); auth is now enforced in the endpoint. Reset PIN compared constant-time
- [x] **Reset/scan race** — `/api/reset` now takes the scan lock; previously an in-flight scan's end-of-scan persist could resurrect the just-cleared portfolio state
- [x] **Peak equity ratchet** — `peak_equity` now updates on mark-to-market, not only on closes, so drawdown mode sees peaks reached while positions were open
- [x] **Tests** — 229 passing (was 206): trailing-stop labeling, position/cash caps, currency-safe ATR check, VIX-halt holdings sweep, FX-guard semantics, US DST market hours, heuristic rate-limiting/grace period

### Learning-loop completion (2026-07-02)
- [x] **MIPRO counterfactual training data** — PASS decisions now persist their decision-time price + exact DSPy inputs (one blob per track/ticker/day to keep the Pi DB small; `decisions` table migrated in-place via `ALTER TABLE`). At MIPRO time, `_build_counterfactual_examples` labels mature PASSes from what the price actually did over `counterfactual_horizon_days` (14d): forward return ≥ 3% → missed BUY, ≤ 0 → correct PASS, ambiguous middle skipped. Counterfactuals are capped at the number of real-trade examples and the 80/20 split is seeded-shuffled so the val set isn't purely hindsight-labeled. This removes the survivorship bias where the optimizer only ever saw taken trades
- [x] **Heuristic outcome feedback** — positions carry the `heuristic_ids` used at entry (outside the DSPy signature); on close, `HeuristicStore.record_outcome` moves each heuristic's `quality_score` by up to ±1 (pnl-scaled, clamped 0–10) and tracks `outcome_count`/`cumulative_pnl_pct`, so validated rules rise and repeatedly harmful ones drift into prune range regardless of the model's initial self-assessment
- [x] **Nordic news prefilter** — `_prefilter` now matches the company name from `universe.csv` ("Volvo" for VOLV-B.ST, share-class suffix stripped), not just the ticker base that never appears in headlines
- [x] **Tests** — 250 passing: counterfactual labeling/horizon/cap/track isolation, decision-persistence dedupe, in-place DB migration, outcome scoring bounds, close-hook wiring, prefilter name matching

### Backtester realism + counterfactual paths + Hurst flag (2026-07-02)
- [x] **Backtester mirrors live execution** — slippage + commissions from settings (FX fee on US), ATR-scaled trailing stop with `trailing_stop` exit labeling, intraday High/Low stop/target checks (stop-first when both trade in one bar, gaps fill at the open), mark-to-market equity so drawdown mode sees open losses; Sharpe annualized by actual holding period; metrics report net-of-commission P&L + `total_commission`. No look-ahead: the trailing stop is raised from a bar's close only *after* that bar's exit checks
- [x] **Counterfactual path simulation** — PASS decisions also persist decision-time ATR; the counterfactual builder simulates the trade the system would have taken (1.5×ATR stop, RRR-2.0 target, stop-first) through the forward OHLC window, so a rally that would have traded through its stop first labels as a correct PASS, not a missed BUY. Falls back to horizon-close labeling when ATR/High/Low are unavailable
- [x] **Hurst on returns (opt-in)** — proper windowed R/S on log returns behind `hurst_on_returns` (default **off**). The returns estimator measures persistence correctly, but a plain drifting random walk then reads ~0.5 (neutral) and the screener gets much stricter — flip deliberately on the Pi and observe candidate volume before committing
- [x] **Tests** — 271 passing: intraday exits/gap fills, backtest trailing + no-look-ahead, cost arithmetic, mark-to-market drawdown, path-simulation labels (stop-first rally case), AR(1) persistent/anti-persistent Hurst on returns

### Correlation cap + schema cleanup (2026-07-02)
- [x] **Pairwise return-correlation cap** — the 0.7 max-correlation rule is now enforced: at risk validation, the candidate's 60-day daily returns are correlated against each same-market open position using the batch OHLCV already fetched (no extra network); any pair above `max_sector_correlation` (0.7) rejects the entry with the offending ticker named. Applied identically in the backtester (on the no-look-ahead slices). Cross-market pairs are skipped — bars don't align and different sessions mute correlation anyway
- [x] **Dead DB tables dropped** — `Trade`, `Position`, `PortfolioSnapshot`, `Heuristic` model classes removed (never written; live state is `portfolio_state`, heuristics are file-backed, `decisions` is the audit trail). Empty tables in existing Pi DBs are harmless leftovers
- [x] **systemd service paths fixed** — `/home/pi/DeepSwing` → `/home/alexander/Documents/DeepSwing`, `User=alexander`
- [x] **Tests** — 285 passing: correlation math (identical/inverse/independent series, overlap minimum, never-raises guard), risk-cap rejection/allowance/worst-pair selection

### Pre-deploy ops hardening (2026-07-02)
- [x] **Nightly SQLite snapshot** — 23:45 CET, SQLite online-backup API (torn-write-safe) into `data/backups/`, newest `db_backup_keep` (7) kept; the portfolio DB previously had no backup at all on the SD card
- [x] **Decisions retention** — weekly maintenance prunes decision rows older than `decisions_retention_days` (90); the table otherwise grows ~1k rows/day forever
- [x] **`.env.example` synced** — documents all knobs added this cycle (position cap, trailing multiplier, `hurst_on_returns`, counterfactual tuning, retention/backup)
- [x] **Dashboard heuristic cards** show outcome feedback (trades used + cumulative P&L) next to quality/usage
- [x] **ARCHITECTURE.md de-staled** — current model IDs, BUY/PASS signature, real screener thresholds, ATR trailing stop, US hours in ET, correlation cap
- [x] **Tests** — 289 passing: retention pruning, snapshot creation/rotation/validity, disabled modes

### Target discipline (2026-07-02)
- [x] **`_fix_rrr` removed** — weak-target BUYs (RRR < 2.0) are rejected by risk validation instead of silently stretched, so the optimizer sees the model's real target placement. Risk-BLOCKED BUYs persist their price/ATR/inputs and feed the counterfactual pipeline like PASSes, so the learning volume that stretching used to provide is preserved without taking the trades

### Offsite backup (2026-07-02) — after an SD-card corruption wiped the Pi
- [x] **rclone → Google Drive nightly backup** — `deploy/backup_to_gdrive.sh` snapshots the DB (SQLite online-backup API), heuristics, compiled programs, and optionally `.env` into one archive and pushes it to a cloud remote, keeping the newest `BACKUP_KEEP` (14). Runs as an **independent** systemd timer (`deepswing-backup.{service,timer}`, nightly 23:50) so it survives an app crash — the app's own `data/backups/` snapshots live on the same card and did NOT protect against card death
- [x] **One-command restore** — `deploy/restore_from_gdrive.sh` pulls the newest (or a named) archive and drops the DB/heuristics/compiled/.env back into place on a fresh Pi
- [x] **Docs** — SETUP.md §4b walks through rclone setup (incl. headless auth), the `/etc/default/deepswing-backup` env file, timer install, an immediate verification run, and the restore procedure

---

## To Do 🔲

### Improvements
- [ ] **Flip `hurst_on_returns`** — the returns-based estimator is implemented and tested but defaults off; enable on the Pi, watch screener candidate volume for a week, then commit or revert
- [ ] **News model on reasoning tier** — `gpt-5-mini` may spend budget on reasoning; monitor Swedish news summary quality, bump model or tune `max_completion_tokens` if weak

### Pi Deployment / Ops
- [ ] Verify APScheduler fires correctly across DST changes (Stockholm CET↔CEST)
- [ ] Monitor memory usage during the first weekly MIPRO run (Pi 5, 1G cap)
- [ ] Complete the MIPRO backup repo setup on the Pi (`MIPRO_BACKUP_REPO_DIR`) before the first MIPRO run
- [ ] Reinstall `systemd/deepswing.service` on the Pi (paths now corrected in-repo: `cp systemd/deepswing.service /etc/systemd/system/ && systemctl daemon-reload`)

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
| Reasoning-model IDs | GPT-5/5.5 and Claude 5 IDs are env-overridable; a wrong ID surfaces at boot via preflight but still requires a manual `.env` fix |
| Non-SEK FX unavailable | If an FX rate can't be resolved, entries are blocked and price updates skipped (never booked raw); a persistent FX outage means stops on non-SEK holdings don't advance until rates return |
| Dashboard sessions | Session tokens are in-memory; a process restart logs all dashboard users out (they just log in again) |
