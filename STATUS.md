# DeepSwing — Implementation Status

Last updated: 2026-08-21

---

## Done ✅

### The objective had almost nothing to optimise against (2026-08-30)
First real run of the replay harness (395 examples, 129 tickers, three batch
fetches) passed its gate — oracle 0.5484 > always-pass 0.5000 > always-buy
0.4625 — and immediately showed something the live system never could:
**perfect foresight beat doing nothing by only 0.052.**

81% of the corpus are non-payers, and a correct PASS scored a flat 0.5, so the
best achievable score on four-fifths of the data was exactly the do-nothing
score. Every point of the oracle's edge came from the 19% that paid, while the
expectancy problem lives in the losers. MIPRO would have been selecting
instruction sets inside a 0.052 band, on a validation split, against noise.

A PASS now earns the R it avoided. Verified on the real distribution: oracle
0.6403, always-pass 0.5363, always-buy 0.4637 — **headroom 0.1041**, double.
The harness reports that gap directly, since a flat 0.5 baseline no longer
exists to read results against.

This does not fix the inertia degenerate: always-pass still beats always-buy on
a corpus with no edge, and should. What it fixes is the oracle pulling clearly
ahead of both.

MIPRO's counterfactual rows now go through the same `select_decision_rows`
sampler the harness uses (frequency-ordered, ≤5 per ticker). The composition
cap — hindsight vs lived, `counterfactual_ratio_cap` — is a separate concern
and deliberately unchanged: at 30 real trades, letting all ~395 counterfactuals
in would be a 13:1 ratio and would undo it.


### Prompt evaluation no longer requires a month of trading (2026-08-30)
The only way to compare two prompts was to trade each for ~30 closed trades.
The backtester cannot help — it contains no model at all and buys every
candidate surviving the screener and risk check — and `metrics_by_program` can
only compare programs that already ran sequentially against different markets.

`src/agent/replay.py` closes it with machinery that already existed:
`Decision.entry_inputs` stores exactly the five DSPy fields, and replay is the
same `program(**entry_inputs)` call the live path makes; `_label_forward_path`
supplies ground-truth R identical to the MIPRO trainset. Corpus building is
network-bound and cached to JSON; scoring reads the cache, so comparing
programs costs no further price data.

`scripts/replay_decisions.py --reference` scores oracle / always-buy /
always-pass using no LLM at all — the harness validating itself. On a corpus
with real edge those must order oracle > always-buy > always-pass; if they
don't, no verdict it gives about a real prompt is worth anything. `always-pass`
scores exactly 0.500 on any corpus, so the report carries totalR, precision and
recall alongside the metric to separate correct caution from inertia.

`decision_metric` moved from `optimizer.py` to `metrics.py` so the scoring path
imports no dspy; `optimizer._pnl_weighted_metric` is now an alias, keeping MIPRO
and the harness on one scoring path by construction.

**Build the first corpus from the pre-reset backup** — the reset deleted 5,551
decision rows that are already past the counterfactual horizon, and the live DB
will have nothing labellable for two weeks.


### Book capacity was the real ceiling on learning (2026-08-30)
`max_position_pct = 0.25` against `market_allocation` (nordic 0.4 / eu 0.2 / us 0.4)
pinned the book at **two concurrent positions** — and a 25% position could not fit
the 20% EU budget at all, so **EU was structurally untradeable**, zero entries
possible, nothing logging it. At ~9-day holds that was **135 days** to accumulate
the 30 trades MIPRO needs.

`max_position_pct` → 0.10 gives 10 slots (nordic 4 / eu 2 / us 4), ~33 trades per
month per track, **~27 days** to 30 trades. `max_risk_per_trade` reverted 1.5% →
1% since the value cap now binds nearly everywhere and 1.5% misstated what the
system actually risks (effective risk is ~0.3–0.9% of equity).

The trade is deliberate: a flatter equity curve and a `drawdown_pause_threshold`
that rarely trips, in exchange for 5× the learning rate. R is invariant to
position size, so the MIPRO metric, `rrr_achieved`, ERL and heuristic scoring all
see identical signal.

Counterfactuals were capped at parity with real trades — a live run discarded 60
of 90 available labelled PASS decisions and left MIPRO picking instructions on a
**12-example** validation split. Now capped at `counterfactual_ratio_cap` (4×),
giving 150 examples and a 30-example split at the MIPRO threshold.

**Watch:** with 10 concurrent positions, `max_positions_per_sector = 2` needs five
distinct sectors to fill the book and the 0.7 correlation cap will bind more
often. If those become the new ceiling, that is the next knob — not position size.


### MIPRO could never actually compile (2026-08-30)
The 2026-08-30 02:00 run reached the end of `MIPROv2.compile()` on the gpt
track — 35 trades, 30 counterfactuals, demos bootstrapped, three candidate
instruction sets proposed, roughly seven minutes of heavy prompt-model calls —
and then died on `ImportError: MIPROv2 requires optional dependency 'optuna'`.
`optuna` was never in `requirements.txt`; `dspy-ai` does not pull it in, and
MIPROv2 imports it at the *end* of compile, in `_optimize_prompt_parameters`.

So no compiled program has ever existed on either track (confirmed at reset:
`programs_archived: 0` for both, empty `compiled/`), and every weekly attempt
past the trade threshold would have paid full proposer cost for nothing.

Added `optuna>=3.6.0` to requirements, plus a guard at the top of
`run_mipro_optimization` that checks importability before any LM is built, so
a missing backend costs one log line rather than a proposer run.


### Heuristic quality for the learning loop (2026-08-30)
Four structural problems in what feeds MIPRO, none of them sample size:

- **Heuristics were scored on one side of their own influence.** `heuristic_ids`
  rode only on opened positions, so a rule that argued for passing was never
  credited when passing was right nor charged when it cost a winner. They now
  ride on PASS/BLOCKED decisions too, and `score_heuristics_from_decisions`
  scores aged ones from their counterfactual forward label (sign follows the
  decision the rules informed). Idempotent via `decisions.heuristics_scored`.
- **The MIPRO metric could not see the right tail.** `tanh(pnl_pct × 10)` put a
  +30% trade 0.045 above a +15% one — the re-tune for bigger wins would have
  been invisible to the optimizer. Now scores R-multiples at scale 0.35, where
  2.5R and 5R differ by 0.12.
- **Core status was popularity, not performance.** `promote_core` checked only
  access count — how often a rule was *shown* — and granted a permanent +2.0
  retrieval boost with no demotion. Now requires quality too, and revokes it.
- **Single-trade rules ranked alongside proven ones.** ERL extracts one
  heuristic per trade; new ones now carry a retrieval penalty until
  `MIN_CORROBORATION` outcomes. Ranked down, not excluded — otherwise they
  could never earn the outcomes that would prove them.

Also made `rrr_achieved` net of commission, matching `pnl_pct`, `win_rate` and
the equity curve. A full stop-out reads ≈ −1.05R rather than −1.00R.

**Known and not fixed:** PASS scores exactly 0.5 on every example, so on a
losing trainset the do-nothing program wins on the real-trade half.
Counterfactual missed-BUY examples are the only counterweight, and they are
capped at the real-trade count. Worth watching on the gpt track.


### Breakeven floor + honest exit labels (2026-08-30)
Live data showed `trailing_stop` exits averaging **−0.52R over 8 trades
(claude)** and **−0.32R over 16 (gpt)** against `take_profit` at +2.46R/+3.44R
— 24 of 56 closed trades exiting at a loss under a winning label, consuming the
entire expectancy margin of the claude track.

Cause: `trail_distance` (2.0×ATR) is always wider than the maximum permitted
stop (1.65×ATR), so the ratchet clears `stop_loss` at ≈ entry+0.35 ATR while
still sitting at a full-risk loss, and only guarantees breakeven at +2.0 ATR.

Fix: an independent `breakeven_armed` floor on `OpenPosition` (arms at
`breakeven_arm_atr_multiplier`, net of both commission legs and exit slippage,
never retreats) plus `resolve_exit()`, which labels by the level that actually
bound — `trailing_stop` only above entry, new `breakeven_stop` for the floor,
`stop_loss` otherwise. Shared between the live simulator and the backtester;
the backtester arms from the close *after* its exit checks so no-look-ahead
holds. Also fixed the ERL fallback label, which narrated an empty
`exit_reason` as a success-flavoured "manual/target exit".

Note the floor rescues a trade only if its peak cleared the arming threshold;
one that reversed from +0.6 ATR still books ≈−1R, but is now honestly labelled
`stop_loss` so ERL attributes it to entry quality.

### Re-tune for move-scaled payoffs (2026-08-30)
A winner pays `risk × RRR`, independent of ATR or target distance, and both
levers were suppressed. `min_rrr` 2.0 → 2.5; the `TradeDecision` docstring now
places stop/target from structure with the numeric floor moved to the
output-field description (three restatements plus a worked example at exactly
2.0 anchored every target at the minimum). The screener never referenced
`atr_14` at all and its RSI curve peaked at 52.5 while paying zero by 67.5 —
added `min_atr_pct` (2%) plus a range score term, re-centred RSI on 60, raised
`rsi_max` to 78. `max_risk_per_trade` 1% → 1.5%. `RiskValidation.size_scale`
now reports when the 25% value cap shrank a position, which previously bound
invisibly. Claude decision budget 1024 → 4096 tokens.

Held tickers are dropped before the decision loop instead of after a news fetch
and one decision call per track. Counterfactual labelling mirrors the breakeven
floor, so hindsight and lived examples share an exit policy.

`/api/reset` now archives the compiled MIPRO program — it previously survived a
reset and kept driving entries from the wiped book. `scripts/purge_dead_tracks.py`
clears the inert options/Kalshi rows that `/api/reset` cannot target.


### Options tracks — removed (2026-08-21)
The `claude-opt` / `gpt-opt` options tracks were shut down and deleted. Gone:
`options_math`, `options_chain`, `options_decision`, `options_risk`,
`options_simulator`, `options_scan`, `vol_context`, `OPTIONS_TRACK.md`,
`tests/test_options.py`, `run_options_mipro`, the bearish screener mirror
(`screen_bearish_candidates`), the triage `sides` argument, every `options_*`
setting, and the dashboard's options tabs/columns/scan button. DeepSwing is
long-only equities on the `claude` / `gpt` tracks again.

Leftover state is inert but not auto-purged — a Pi with options history still
carries `portfolio_state` rows and `decisions` rows for `claude-opt`/`gpt-opt`,
`heuristics/claude-opt/` + `heuristics/gpt-opt/`, and
`compiled/*_option_decision*.json`. Nothing reads them; delete when convenient.

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
- [x] `src/agent/erl.py` — post-trade causal analysis; Claude Opus + extended thinking (Claude); GPT-5.6-sol + `reasoning_effort` (GPT); structured heuristic extraction + storage
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
- [x] **Network watchdog** — the Pi's Wi-Fi dropped twice on 2026-07-13 (tunnels + SSH dark, box fine); `deploy/net-watchdog.sh` + systemd timer bounces the interface and reboots after 3 failed recoveries; SETUP.md §8 covers install + disabling Wi-Fi power save
- [x] **Pre-decision triage** — `src/analysis/triage.py`: one cheap shared `gpt-5-mini` call ranks the screener's candidates and only `triage_keep_top` (5) reach news + per-track decisions; fails open to screener top-K; cuts the dominant per-scan LLM cost by ~2/3 (`tests/test_triage.py`)

### Correctness & security review fixes (2026-07-02)
- [x] **VIX halt no longer abandons holdings** — a VIX ≥ 35 halt blocks new entries but falls through to the holdings monitor, so stops/targets/news exits still run during volatility spikes
- [x] **ATR-scaled trailing stop + correct exit labels** — the fixed 2% trail (tighter than most tickers' daily ATR; killed winners long before the RRR 2.0 target) is now `trailing_stop_atr_multiplier` (2×ATR, SEK-converted at entry, persisted per position); trailed exits close as `exit_reason="trailing_stop"` instead of being mislabeled `"stop_loss"`. **Superseded 2026-08-30** — that predicate ("did the trail ratchet above stop_loss") was true while the trail was still below entry, so it inverted the error: ≈−1R losses were labelled trailed winners. See the breakeven-floor entry below.
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
- [ ] **Backtest the re-tune before trusting it** — `POST /api/backtest` on a fixed window, comparing expectancy, exit-reason mix and the realised RRR distribution against the pre-change parameters. If `min_atr_pct` + RRR 2.5 collapse trade count rather than widening the right tail, back them off there rather than live
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
| Reasoning-model IDs | GPT-5/5.6-sol and Claude 5 IDs are env-overridable; a wrong ID surfaces at boot via preflight but still requires a manual `.env` fix |
| Non-SEK FX unavailable | If an FX rate can't be resolved, entries are blocked and price updates skipped (never booked raw); a persistent FX outage means stops on non-SEK holdings don't advance until rates return |
| Dashboard sessions | Session tokens are in-memory; a process restart logs all dashboard users out (they just log in again) |
