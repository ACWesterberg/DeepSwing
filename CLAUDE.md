# DeepSwing — Claude Code Context

This file gives Claude Code enough context to resume work on this project in any session (CLI, web, or mobile).

---

## What this project is

An AI-powered **swing trading simulator** running on a Raspberry Pi 5. Paper-trading only (no real money). Two parallel simulation tracks — **Claude** and **GPT** — make independent trading decisions on the same market data so their performance can be compared. Prompts evolve over time via DSPy/MIPRO optimization. The system learns from closed trades via Experiential Reflective Learning (ERL), extracting reusable heuristics.

Two additional **options tracks** — `claude-opt` and `gpt-opt` — trade long single-leg US call options on the same candidate pipeline (yfinance chains, premium-based risk, own MIPRO-optimized prompt). See [OPTIONS_TRACK.md](OPTIONS_TRACK.md) for the design.

---

## Key design decisions (don't re-litigate these)

- **No FinBERT** — Claude Haiku handles news analysis; it understands Swedish, provides per-ticker reasoning, not just sentiment labels
- **Thinking models only for ERL** — standard models for 15-min scan decisions (latency + cost); Claude Sonnet with `extended_thinking=True` for post-trade causal analysis (deeper reasoning, async)
- **`ta` library, not `pandas-ta`** — pandas-ta requires Python 3.12+; `ta` covers all needed indicators and is Pi-safe
- **DSPy 2.6 uses `dspy.configure(lm=...)`** — not `with dspy.settings.context(lm=...)` (deprecated in 2.6+)
- **All DB records have a `track` column** — "claude" | "gpt"; heuristics stored in `heuristics/{track}/`
- **MIPRO runs weekly, Sunday 02:00 CET** — requires 30+ closed trades to run; archives previous compiled JSON
- **Capacity-aware scanning** — a track with free cash below `min_cash_for_new_position_pct` (5%) of its equity is treated as fully allocated and gets no entry decisions; when *no* track is funded the scan skips the whole candidate/news/decision pipeline and runs a holdings-only monitor.
- **Per-market allocation cap** — each market's open-position value is capped at `market_allocation[market]` (default 0.5 for both `nordic` and `us`) of a track's equity, so the long US session can't fill the whole book before Stockholm opens. Enforced twice: `Portfolio.can_open_in_market` gates the scan (falls through to the holdings monitor when a market's budget is exhausted), and sizing passes `market_budget_remaining` as `available_cash` so a single scan can't overshoot the cap. A market omitted from the dict, or set ≥ 1.0, is cash-limited only. Holdings are tracked on price alone — a news pull + AI exit review only fires once a position moves ≥ `holdings_news_jump_pct` (5%) since its last check (closes as `exit_reason="news_exit"`). Set either knob to 0 to restore always-on behaviour.
- **Portfolio state is durable** — the live `Portfolio` (cash, open positions, closed trades, peak equity) is an in-memory object mirrored to the `portfolio_state` DB table on every open/close and at end of scan, and rehydrated on startup (`persistence.restore_portfolios()`), so tracks survive a redeploy/restart. `main.py` restores *before* arming the persistence handler; `/api/reset` deletes the persisted rows so a restart doesn't resurrect cleared tracks. Heuristics stay file-backed; MIPRO programs stay git-backed.
- **Scans never block the event loop** — `run_scan` is long/blocking (network + LLM), so `/api/scan` offloads it via `run_in_executor`; a module-level `_scan_lock` serializes scans so a manual trigger can't overlap the scheduled one and double-open. The scheduler already runs scans in its own thread.
- **NewsAPI is rate-limit-guarded** — per-ticker news is cached for `news_refresh_interval_minutes`; if a fetch stalls beyond `newsapi_slow_threshold_seconds` (429 backoff) a breaker skips NewsAPI (RSS only) for `newsapi_cooldown_minutes`, so one throttled ticker doesn't cost ~1 min each. The jump-triggered exit review passes `force_refresh=True` for freshness.
- **Per-ticker news has a free fallback** — when NewsAPI/RSS returns nothing (common for US, which has no RSS), `fetch_news_for_ticker` falls back to a free source so US tickers still get news: yfinance/Yahoo (no key, universal backstop), with Finnhub preferred for US when `finnhub_api_key` is set (dormant drop-in until then).
- **The Pi self-heals its network** — Wi-Fi association drops take down SSH + all Cloudflare tunnels while the box runs fine; a 2-min systemd timer (`net-watchdog`) pings the gateway, bounces the interface on failure, and reboots after 3 failed recoveries (safe: portfolio state is DB-backed). Wi-Fi power save is disabled via NetworkManager conf. See SETUP.md §8.
- **Screened candidates are triaged before the expensive models** — every candidate that reaches the decision loop costs a news fetch + news analysis + one decision call per funded track (the dominant LLM spend). One cheap shared call (`triage_model`, default `gpt-5-mini` — same shared-call pattern as news analysis) ranks the screener's output on a one-line technical digest and only `triage_keep_top` (5) proceed; both tracks see the identical surviving set so the head-to-head stays fair. Applies to stock scans and the options scan (calls+puts triaged jointly). Fails open to the screener's own top-K ordering; `triage_keep_top=0` or `triage_enabled=false` disables. `src/analysis/triage.py`.
- **Volume is screened on the last *completed* daily bar** — intraday the latest bar is still forming, so `volume_ratio` from it reads ~0.1× and the `volume_spike_multiplier` gate would reject everything until near the close. `technical.py` computes the ratio from the previous full day vs its trailing 20-day average; `current_volume` still reports the live bar for display.
- **Options tracks are US-only, long single-leg only** — yfinance has no `.ST` chains and free Nordic derivatives data doesn't exist; short options don't fit the risk philosophy. Direction is the screener's job: bullish setups get call shortlists, mirrored bearish setups (`screen_bearish_candidates` — below 50 SMA, RSI 30–65, EMA21<SMA50 or bb%B≥0.65, same volume gate) get put shortlists; `options_enable_puts=False` restores calls-only. The stock tracks never see bearish candidates. Options capital is 1M SEK (vs 100k stocks) because one contract is the minimum lot — at 100k the 1%/2% premium caps blocked essentially every liquid contract; `restore_portfolios` auto-rebases an *untraded* options track to the current capital. The LLM picks a contract **by index from a pre-filtered shortlist** (21–60 DTE, |delta| 0.35–0.65, OI ≥ 200, spread ≤ 8%, ATR·√DTE expected move must cover the breakeven distance ≥ `options_min_move_coverage`) — it never sees a raw chain and can't invent contracts. Each line carries breakeven + move-coverage math, and a separate `volatility_context` signature input gives realized-vol percentile and ATM-IV-vs-realized pricing (free-data IV-rank proxy — no IV history exists, so IV is ranked against the year's realized-vol range). Exits are premium-relative and mechanical: profit target %, premium stop % (both model-chosen, clamped), a time stop in DTE, and a daily 22:10 CET expiry sweep that settles at intrinsic. Premium paid IS the risk, so sizing caps premium at 1% of equity (2% hard cap for a single contract). Fills at mid + adverse half-spread; flat per-contract commission. Options scans run hourly (not every 15 min) and share `_scan_lock` with stock scans. Track names `claude-opt`/`gpt-opt` reuse all track-keyed plumbing (heuristics dirs, `portfolio_state`, `compiled/{track}_option_decision.json`, ERL provider chosen by `track.startswith("claude")`).

---

## Markets

| Market | Session (CET) | Watchlist | Data Source |
|---|---|---|---|
| Nordic (OMXS30) | 09:00–17:30 | 30 stocks, `.STO` suffix | Alpha Vantage (primary), yfinance `.ST` (fallback) |
| US (NYSE/NASDAQ) | 15:30–22:00 | Top 100 S&P 500 | yfinance |

Both configurable in `config/settings.py` (`nordic_watchlist`, `us_watchlist`).

---

## Models used

| Task | Claude track | GPT track |
|---|---|---|
| Scan decisions (15-min) | `claude-sonnet-5` | `gpt-5` |
| ERL causal analysis | `claude-opus-4-8` + extended thinking | `gpt-5.6-sol` + `reasoning_effort=high` |
| News analysis | `gpt-5-mini` (shared by both tracks) | `gpt-5-mini` |
| Candidate triage (pre-decision filter) | `gpt-5-mini` (shared by both tracks) | `gpt-5-mini` |
| MIPRO — task model (evaluates candidates) | `claude-sonnet-5` | `gpt-5` |
| MIPRO — prompt model (writes instructions) | `claude-opus-4-8` | `gpt-5.6-sol` |

All model IDs are env-overridable (see `.env.example`). Scan/ERL models were upgraded from the original Haiku/4o-mini/Sonnet-4-6/4o tier. The options tracks (`claude-opt`/`gpt-opt`) reuse the same per-provider models for decisions, ERL, and MIPRO — only the DSPy signature and compiled program differ. News analysis is a single shared GPT call (`gpt-5-mini`) fed identically to both tracks — kept on a light model, and on GPT to use the free-token quota. MIPRO uses a heavy proposer (`prompt_model`) to write candidate instructions while the cheaper decision model evaluates them.

---

## Risk rules (all enforced in `src/agent/risk.py` unless noted)

- 1% max risk per trade (hard cap 2%)
- Min RRR 2.0
- Stop-loss at 1.5× ATR below entry — validated as *fractions of price* so the check is currency-safe (entry/stop are SEK, ATR is native currency)
- Position value capped at `max_position_pct` (25%) of equity **and** at available cash — risk-based sizing alone is unbounded when stops are tight
- >10% portfolio drawdown → halve all position sizes
- No duplicate tickers across open positions; max 2 positions per sector
- Pairwise return-correlation cap: candidate vs each same-market open position (60-day daily returns from the scan's batch OHLCV); any pair > `max_sector_correlation` (0.7) rejects the entry — same rule in the backtester
- Trailing stop trails at `trailing_stop_atr_multiplier` (2×) ATR once in profit (`simulator.py`); trailed exits close as `exit_reason="trailing_stop"`, not `"stop_loss"` — ERL depends on this distinction
- VIX ≥ 35 halts **new entries only** — open holdings still get the stop/target sweep and news-exit review (`scan_loop.py` falls through to the holdings monitor)
- Non-SEK prices are never booked without FX conversion — `_to_sek_price` returns `None` on failure and callers skip; never fall back to raw native prices
- US market hours are evaluated in **US Eastern Time** (`market_hours.py`), not fixed CET — the US/EU DST transitions are weeks apart

---

## File map (key files)

```
config/settings.py          All config — API keys, risk params, model names, watchlists
src/db.py                   SQLAlchemy models (PortfolioState, Decision) + in-place SQLite migrations
src/portfolio/persistence.py  DB save/restore of live portfolio state (survives restarts)
src/data/market_data.py     OHLCV fetch — yfinance + Alpha Vantage
src/data/news_fetcher.py    NewsAPI + Swedish RSS; yfinance/Finnhub fallback + rate-limit breaker
src/data/insider_fetcher.py SEC EDGAR + FI Insynsregistret
src/data/macro_data.py      FRED + Riksbank + ECB
src/analysis/technical.py   11 indicators via `ta` library
src/analysis/options_math.py  Black-Scholes price/delta/theta (closed-form, no scipy)
src/data/options_chain.py   yfinance US option chains → liquidity-filtered shortlist
src/agent/options_decision.py  DSPy OptionTradeDecision; per-track engine (claude-opt/gpt-opt)
src/agent/options_risk.py   Premium-budget sizing + contract validation
src/portfolio/options_simulator.py  OptionsPortfolio — long calls, premium stops, expiry
src/scheduler/options_scan.py  Hourly options scan + daily expiry sweep (22:10 CET)
src/analysis/regime.py      Hurst Exponent + autocorrelation → trending/mean-reverting
src/analysis/screener.py    Multi-factor filter → top-N candidates
src/agent/decision.py       DSPy TradeDecision program; DecisionEngine per track
src/agent/risk.py           Position sizing, stop validation, RRR check
src/agent/memory.py         HeuristicStore — file-backed, track-namespaced
src/agent/erl.py            Post-trade causal analysis → heuristic extraction
src/agent/news_analyzer.py  Shared per-ticker news analysis (gpt-5-mini, both tracks)
src/portfolio/simulator.py  Paper trading engine (Portfolio class); dual-track
src/portfolio/metrics.py    Sharpe, drawdown, win rate, MIPRO metric
src/scheduler/market_hours.py  is_market_open(), active_markets(); Nordic in CET, US in ET
src/scheduler/scan_loop.py  Main 15-min cycle: fetch → analyze → screen → decide → trade
src/scheduler/optimizer.py  Weekly MIPROv2 + heuristic prune/promote
src/dashboard/app.py        FastAPI + WebSocket; /api/comparison is the key endpoint
src/dashboard/static/app.js Chart.js equity curves, head-to-head table, auto-refresh
main.py                     Entry point: DB init + APScheduler + uvicorn
```

---

## Running locally

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env    # add API keys
venv/bin/python main.py
# dashboard at http://localhost:8000
```

Manual scan (no need to wait for scheduler):
```bash
curl -X POST http://localhost:8000/api/scan/nordic
curl -X POST http://localhost:8000/api/scan/us
```

---

## Learning loop (how the system improves)

- **MIPRO trainset** = real closed trades (labeled by realized P&L) **plus counterfactuals**: PASS decisions persist their DSPy inputs + decision-time price (one blob per track/ticker/day); at MIPRO time they're labeled from the forward return over `counterfactual_horizon_days` (≥3% → missed BUY, ≤0 → correct PASS, middle skipped). Counterfactuals are capped at the real-trade count.
- **Heuristic feedback**: positions carry `heuristic_ids` in `entry_inputs` (added in `scan_loop`, *never* passed into the DSPy program call); on close `record_outcome` moves quality by up to ±1 pnl-scaled, clamped 0–10. Access counts increment at most once/hour; prune has a 7-day grace period.
- **News prefilter** matches the company name from `universe.csv` (headlines say "Volvo", never "VOLV-B").

---

## What's left to build

See [STATUS.md](STATUS.md) for the full To Do list. Priority items:

1. **Flip `hurst_on_returns`** — the returns-based R/S estimator is implemented behind a settings flag (default off, because it reclassifies drifting walks as neutral and makes the screener stricter); enable deliberately and observe candidate volume
2. **News summary quality** — monitor whether `gpt-5-mini` spends its budget on reasoning at the expense of the Swedish summaries

There is **no target auto-stretching**: a BUY whose own target gives RRR < 2.0 is rejected at risk validation and learned from counterfactually (blocked BUYs persist inputs like PASSes). Don't reintroduce `_fix_rrr`.

The backtester now mirrors live execution (slippage/commissions, intraday High/Low exits, ATR trailing stop, mark-to-market equity, correlation cap); counterfactual labels simulate the stop/target path when ATR is available.

---

## Style conventions

- No comments unless the WHY is non-obvious
- No docstrings longer than one line
- Trust imports — don't add redundant `try/except` around internal calls
- Type hints on all function signatures
- `from __future__ import annotations` at top of every file
