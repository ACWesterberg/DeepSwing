# DeepSwing — Implementation Status

Last updated: 2026-06-26

---

## Done ✅

### Phase 1 — Foundation
- [x] Project scaffolding, directory structure, `__init__.py` files
- [x] `requirements.txt` (Python 3.11, all Pi-safe dependencies)
- [x] `.env.example` with all required API key slots
- [x] `config/settings.py` — Pydantic Settings, dual-track config, risk params, watchlists
- [x] `src/db.py` — SQLAlchemy models: Trade, Position, PortfolioSnapshot, Heuristic (all with `track` column)
- [x] `src/data/market_data.py` — yfinance (US + Nordic fallback) + Alpha Vantage primary for Nordic; batch fetchers; 4h cache for AV
- [x] `src/analysis/technical.py` — 11 indicators via `ta` library: EMA/SMA, ATR, Bollinger Bands, RSI, Parabolic SAR, EOM, OBV, Fibonacci
- [x] `src/analysis/regime.py` — Hurst Exponent (R/S analysis) + lag-1 autocorrelation; trending/mean-reverting/neutral classification
- [x] Database init (`init_db()`)

### Phase 2 — Core Agent
- [x] `src/analysis/screener.py` — multi-factor filter (SMA, RSI, volume, regime); weighted scoring; top-N candidates
- [x] `src/agent/risk.py` — ATR-based stop validation, RRR check, 1% position sizing, drawdown mode halving, duplicate ticker check
- [x] `src/agent/memory.py` — file-backed heuristic store; track-namespaced; retrieve by regime/market relevance; prune; promote core rules
- [x] `src/agent/decision.py` — DSPy `TradeDecision` signature; `DecisionEngine` per track; loads compiled program if available, else baseline; `dspy.configure()` per call
- [x] `src/agent/news_analyzer.py` — keyword pre-filter → Claude Haiku for per-ticker contextual analysis (Swedish + English)

### Phase 3 — Simulation + ERL + DSPy Optimization
- [x] `src/portfolio/simulator.py` — track-tagged paper portfolio; open/close with slippage; trailing stop; stop-loss/take-profit auto-close; drawdown mode flag
- [x] `src/portfolio/metrics.py` — Sharpe ratio, max drawdown, win rate, avg RRR, total return, `optimization_metric = win_rate × avg_rrr`
- [x] `src/agent/erl.py` — post-trade causal analysis; Claude Sonnet + extended thinking (Claude track); GPT-4o (GPT track); structured heuristic extraction + storage
- [x] `src/scheduler/optimizer.py` — weekly MIPROv2 optimization per track; archives previous compiled program; calls `DecisionEngine.reload()`; heuristic prune/promote

### Phase 4 — Scheduler + Data Ingestion
- [x] `src/scheduler/market_hours.py` — `is_market_open()`, `active_markets()`, CET timezone, weekday-aware
- [x] `src/scheduler/scan_loop.py` — full scan cycle: data → technicals → regime → screen → (per candidate × per track: heuristics → decision → risk → execute) → position updates → ERL trigger
- [x] `src/data/news_fetcher.py` — NewsAPI (EN) + DI.se/Börsdata/Redeye RSS (SE); deduplication
- [x] `src/data/insider_fetcher.py` — SEC EDGAR Form 4 search (US); FI Insynsregistret CSV (SE); 24h cache
- [x] `src/data/macro_data.py` — FRED (Fed Funds Rate, CPI, 10Y Treasury, Unemployment); Riksbank SWEA API; ECB deposit rate; 6h cache

### Phase 5 — Dashboard
- [x] `src/dashboard/app.py` — FastAPI; REST endpoints: `/api/status`, `/api/portfolio/{track}`, `/api/trades/{track}`, `/api/comparison`, `/api/heuristics/{track}`, `POST /api/scan/{market}`; WebSocket `/ws`
- [x] `src/dashboard/templates/index.html` — tab navigation: Comparison, Claude, GPT, Heuristics (both tracks)
- [x] `src/dashboard/static/style.css` — dark theme, track color-coding (purple=Claude, blue=GPT)
- [x] `src/dashboard/static/app.js` — equity curve overlay chart (Chart.js), head-to-head metrics table, positions/trades tables, heuristic cards, auto-refresh every 60s + WebSocket push
- [x] `main.py` — entry point: DB init, APScheduler (15-min scan + Sunday 02:00 MIPRO), uvicorn
- [x] `systemd/deepswing.service` — autostart on Pi boot, Pi 5 resource limits

### Documentation & Deployment
- [x] `SETUP.md` — Pi 5 hardware, OS setup, Python install, venv, systemd service, Cloudflare Tunnel custom domain, Cloudflare Access auth
- [x] `README.md` — project overview, stack table, quick start, API keys
- [x] `ARCHITECTURE.md` — full system flow, dual-track diagram, DSPy signature, ERL loop, risk rules, data sources
- [x] `STATUS.md` — this file
- [x] `CLAUDE.md` — session context for resuming in Claude Code (web or CLI)
- [x] `.gitignore` — excludes `.env`, `venv/`, `data/*.db`, `heuristics/`, `compiled/`
- [x] GitHub repo created and initial commit pushed

---

## To Do 🔲

### Testing
- [ ] Unit tests for `technical.py` — verify each indicator value against known inputs
- [ ] Unit tests for `regime.py` — Hurst H>0.55 on trending synthetic data, H<0.45 on mean-reverting
- [ ] Unit tests for `screener.py` — confirm filter logic and scoring
- [ ] Unit tests for `risk.py` — position sizing math, RRR rejection cases
- [ ] Integration test for `scan_loop.py` — mock API calls, verify full cycle produces decisions
- [ ] End-to-end test: open trade → price update → stop hit → ERL heuristic file created

### Improvements
- [ ] **Insider data parsing** — FI Insynsregistret CSV column names vary by export version; needs more robust header detection
- [ ] **Sector correlation check** — current implementation only blocks duplicate tickers; needs sector-level correlation matrix (yfinance sector tags) to enforce the 0.7 max correlation rule properly
- [x] **ERL / MIPRO input capture** — trade-entry DSPy inputs (technicals, regime, news, macro, heuristics) are now captured in `decision.py` and stored on `OpenPosition.entry_inputs`, carried to `ClosedTrade`, and consumed by `optimizer.py` to build the MIPRO trainset. Previously `optimizer.py` checked `hasattr(t, "_entry_inputs")` which was never set, so the trainset was always empty and MIPRO always skipped.
- [ ] **Walk-forward validation** — backtesting harness to validate strategy parameters on historical data before deploying changes
- [ ] **OMXS30 dynamic watchlist** — currently hardcoded; fetch from an API or scrape the official composition periodically
- [ ] **VIX/OMXVIX turbulence halt** — fetch VIX as a circuit-breaker for extreme volatility (currently noted as configurable but not implemented)
- [ ] **Drawdown peak tracking** — `_is_drawdown_mode()` in `risk.py` currently returns `False` (placeholder); the `Portfolio.is_drawdown_mode` property works correctly via `portfolio.peak_equity`, but `risk.py` needs to receive this state from the scan loop
- [ ] **Alpha Vantage Nordic batch throttling** — free tier is 25 req/day; current implementation spaces calls 2.5s apart but doesn't count daily usage; add a daily counter with graceful fallback to yfinance when limit reached
- [ ] **WebSocket push on trade events** — currently only `POST /api/scan` triggers a broadcast; scan_loop should push live trade opens/closes

### Pi Deployment
- [ ] Deploy to Pi 5 and test full end-to-end with live market data
- [ ] Verify APScheduler fires correctly across DST changes (Stockholm CET↔CEST)
- [ ] Monitor memory usage during weekly MIPRO optimization
- [ ] Set up Cloudflare Tunnel + custom domain
- [ ] Add Cloudflare Access (email OTP) to protect the dashboard

### After First 30+ Closed Trades
- [ ] Verify first MIPRO optimization run produces valid compiled JSON
- [ ] Compare `optimization_metric` (win_rate × avg_rrr) between pre- and post-MIPRO
- [ ] Review ERL heuristics for quality — are they specific and actionable?
- [ ] Begin tracking Claude vs GPT divergence in decisions on the same candidates

---

## Known Limitations

| Item | Detail |
|---|---|
| Python 3.9 on macOS dev machine | `pandas-ta` requires Python 3.12; switched to `ta` library which covers all needed indicators |
| Alpha Vantage Nordic | Nordic tickers may need the bare symbol (e.g. `ERIC-B` not `ERIC-B.STO`) for the TIME_SERIES_DAILY_ADJUSTED endpoint — test per ticker |
| DSPy 2.6 global LM | `dspy.configure(lm=...)` sets a global LM; parallel scan of both tracks in the same process will race — sequential track processing is currently used |
| No backtesting yet | System runs forward only; no historical paper-trading simulation available |
