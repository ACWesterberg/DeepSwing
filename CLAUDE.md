# DeepSwing — Claude Code Context

This file gives Claude Code enough context to resume work on this project in any session (CLI, web, or mobile).

---

## What this project is

An AI-powered **swing trading simulator** running on a Raspberry Pi 5. Paper-trading only (no real money). Two parallel simulation tracks — **Claude** and **GPT** — make independent trading decisions on the same market data so their performance can be compared. Prompts evolve over time via DSPy/MIPRO optimization. The system learns from closed trades via Experiential Reflective Learning (ERL), extracting reusable heuristics.

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
- **Screened candidates are triaged before the expensive models** — every candidate that reaches the decision loop costs a news fetch + news analysis + one decision call per funded track (the dominant LLM spend). One cheap shared call (`triage_model`, default `gpt-5-mini` — same shared-call pattern as news analysis) ranks the screener's output on a one-line technical digest and only `triage_keep_top` (5) proceed; both tracks see the identical surviving set so the head-to-head stays fair. Fails open to the screener's own top-K ordering; `triage_keep_top=0` or `triage_enabled=false` disables. `src/analysis/triage.py`.
- **Per-trade P&L is net** — slippage lives in the fill prices, commissions (entry + exit legs) are subtracted in `ClosedTrade.pnl`/`pnl_pct`; pre-upgrade persisted trades default to commission 0 (gross). The equity curve, win rate, ERL and heuristic scoring all consume the net figures. `avg_rrr`/`optimization_metric` are the mean R-multiple across **all** trades (expectancy) — never average winners only.
- **Entries fill at a live quote, never the scan-time OHLCV close** — the daily close a candidate was screened on can be hours stale (Alpha Vantage is EOD; EU feeds are delayed), and booking it while exits fill live realized the gap as phantom P&L (0-day ±huge trades). If no live quote exists, or it deviates more than `max_entry_price_deviation` (3%) from the scan price, the entry is blocked. News-exit fills use the same live feed. Entry-side FX resolution is *strict* — a ticker whose quote currency can't be resolved from its suffix/market (e.g. an `.IS` listing in the US watchlist) is blocked instead of booked at a guessed rate; exit pricing of existing positions stays lenient so they can still close consistently.
- **Volume is screened on the last *completed* daily bar** — intraday the latest bar is still forming, so `volume_ratio` from it reads ~0.1× and the `volume_spike_multiplier` gate would reject everything until near the close. `technical.py` computes the ratio from the previous full day vs its trailing 20-day average; `current_volume` still reports the live bar for display.

- **Prediction-market track is a `market`, not a new AI axis** — Kalshi weather contracts run as `market="events"` with their own cash pools (`claude_events` / `gpt_events`), so event P&L never contaminates the Claude-vs-GPT equity comparison while both models still trade the same book. Emptying `settings.event_tracks` disables the whole track — every wiring point is guarded on it, the way the removed options tracks were. The edge is **arithmetic, not the LLM**: an NWS forecast distribution integrated over each strike bucket gives a fair probability, and the model's only job is to veto edges that are stale or artifacts. Never let it produce the probability. Binary contracts have no stop, ATR or RRR, so `risk.py` does not apply — `event_risk.py` sizes with fractional Kelly, and event positions never go through `update_prices()` (its ATR trailing stop would close winners). Kalshi's fee, `ceil(0.07·C·P·(1−P))` cents on entry only, is modelled exactly: at 50¢ it is 3.5% of stake and it is what decides whether a measured edge is real. Fills cross the real bid/ask rather than `simulated_slippage`. Settlement reads Kalshi's own `result` rather than re-deriving the observed high, at the FX rate the position opened at so P&L measures the edge and not USD/SEK drift. **The event date comes from the event ticker (`KXHIGHDEN-26AUG27`), never from `close_time`** — a market stays open past the day it covers while settlement waits on the next morning's climate report, and deriving the day from `close_time` priced yesterday's contract against today's forecast. Two gates exist because the live book produces artifacts the spread check misses: a resting 1c ask with no bid is dust (`min_event_bid`), and any edge above `max_plausible_edge` is treated as a model fault and logged, because no real edge that large exists on a quoted weather market. **The deliverable is the calibration plot on the Events tab, not the equity curve** — predicted-vs-realised frequency reveals whether the forecast model works long before P&L does.

- **Personal watchlist alerts are decoupled from the trading tracks** — the dashboard Watchlist tab stores tickers in `watched_tickers`; a 15-min APScheduler job (`watch_monitor`, never takes `_scan_lock`) pings Telegram on day moves ≥ `watch_move_alert_pct` (3%, re-ping per extra 2% step or direction flip, once-per-day baseline, market-hours only), on fresh directional news, and on insider-summary changes. One shared `gpt-5-mini` call classifies news/insider events bullish/bearish/neutral — **neutral never pings** and the classifier fails closed to neutral. Dedupe state (seen headline hashes, insider hash, last alerted move) lives on the row, so restarts never re-ping; the first pass after adding a ticker baselines silently. Alerts log to `watch_alerts` (dashboard feed, capped at 500) with `delivered=False` when Telegram keys are unset — the whole feature is dormant-but-visible until `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` land in `.env`.

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

All model IDs are env-overridable (see `.env.example`). Scan/ERL models were upgraded from the original Haiku/4o-mini/Sonnet-4-6/4o tier. News analysis is a single shared GPT call (`gpt-5-mini`) fed identically to both tracks — kept on a light model, and on GPT to use the free-token quota. MIPRO uses a heavy proposer (`prompt_model`) to write candidate instructions while the cheaper decision model evaluates them.

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
src/data/kalshi.py          Kalshi public market data (no auth) — UNVERIFIED against live API
src/data/weather_forecast.py  NWS forecasts + AFD discussion — UNVERIFIED against live API
src/analysis/event_model.py    Forecast distribution → fair probability per strike bucket
src/analysis/event_screener.py Edge filter; rejects anything the fee eats
src/agent/event_risk.py     Fractional Kelly + exact Kalshi fee formula
src/agent/event_decision.py DSPy EventTradeDecision — vetoes edges, never forecasts
src/scheduler/event_loop.py Event cycle: settle → price → screen → veto → size → open
scripts/check_event_sources.py  Run on the Pi to verify both APIs before trusting output
src/data/market_data.py     OHLCV fetch — yfinance + Alpha Vantage
src/data/news_fetcher.py    NewsAPI + Swedish RSS; yfinance/Finnhub fallback + rate-limit breaker
src/data/insider_fetcher.py SEC EDGAR + FI Insynsregistret
src/data/macro_data.py      FRED + Riksbank + ECB
src/analysis/technical.py   11 indicators via `ta` library
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
src/scheduler/watch_monitor.py  Personal watchlist: move/news/insider checks → Telegram
src/agent/watch_classifier.py   bullish/bearish/neutral verdicts (gpt-5-mini, fails to neutral)
src/notify/telegram.py      sendMessage wrapper; dormant until bot token + chat id set
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
curl -X POST http://localhost:8000/api/scan/events
```

Before the events track is trusted, verify its two public APIs on the Pi:
```bash
venv/bin/python scripts/check_event_sources.py
```

---

## Learning loop (how the system improves)

- **MIPRO trainset** = real closed trades (labeled by realized P&L) **plus counterfactuals**: PASS decisions persist their DSPy inputs + decision-time price (one blob per track/ticker/day); at MIPRO time they're labeled from the forward return over `counterfactual_horizon_days` (≥3% → missed BUY, ≤0 → correct PASS, middle skipped). Counterfactuals are capped at the real-trade count.
- **Heuristic feedback**: positions carry `heuristic_ids` in `entry_inputs` (added in `scan_loop`, *never* passed into the DSPy program call); on close `record_outcome` moves quality by up to ±1 pnl-scaled, clamped 0–10. Access counts increment at most once/hour; prune has a 7-day grace period.
- **News prefilter** matches the company name from `universe.csv` (headlines say "Volvo", never "VOLV-B"). Universe names are legal names ("Telefonaktiebolaget LM Ericsson (publ)"), so the matcher strips parentheticals + corporate-form words and takes the last distinctive token ("ericsson").

---

## What's left to build

See [STATUS.md](STATUS.md) for the full To Do list. Priority items:

1. **Flip `hurst_on_returns`** — the returns-based R/S estimator is implemented behind a settings flag (default off, because it reclassifies drifting walks as neutral and makes the screener stricter); enable deliberately and observe candidate volume
2. **News summary quality** — monitor whether `gpt-5-mini` spends its budget on reasoning at the expense of the Swedish summaries
3. **Verify the two event APIs on the Pi** — `src/data/kalshi.py` and `src/data/weather_forecast.py` were written from documented schemas without live access. Run `scripts/check_event_sources.py`; it confirms the working Kalshi host, flags any field the parser expects but does not get, and prints the strike types actually in use. **Also confirm each station in `weather_forecast._STATIONS` is the one its Kalshi series resolves on** — a wrong station is a silently biased model, not an error.
4. **Recalibrate `forecast_sigma`** — `forecast_sigma_day1` / `forecast_sigma_per_day` are seeded from published NWS MAE, not measured. Once ~50 contracts have settled, read the calibration plot on the Events tab and fit sigma to observed error before turning `EVENT_DRY_RUN` off.
5. **ERL + MIPRO for event trades** — `run_erl` and the optimizer are equity-shaped (`stop_hit`, RRR, `program_hash` on entries). Event trades currently learn nothing; they need a contract-shaped path before the tracks can improve.

There is **no target auto-stretching**: a BUY whose own target gives RRR < 2.0 is rejected at risk validation and learned from counterfactually (blocked BUYs persist inputs like PASSes). Don't reintroduce `_fix_rrr`.

The backtester now mirrors live execution (slippage/commissions, intraday High/Low exits, ATR trailing stop, mark-to-market equity, correlation cap); counterfactual labels simulate the stop/target path when ATR is available.

---

## Style conventions

- No comments unless the WHY is non-obvious
- No docstrings longer than one line
- Trust imports — don't add redundant `try/except` around internal calls
- Type hints on all function signatures
- `from __future__ import annotations` at top of every file
