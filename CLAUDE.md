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
- **Every OpenAI call sets `reasoning_effort`; none did before.** Reasoning tokens bill as *output* tokens and the API default is `medium`, so the whole GPT side was quietly paying for a thousand-plus invisible tokens per call — on triage (returns a JSON ticker list), news analysis (three sentences) and watch classification (one verdict) as much as on the scan decision. Two tiers: `gpt_decision_reasoning_effort` (default `low`) rides on `build_lm`, so it covers the live scan decision **and** MIPRO's task model, which replays the same program hundreds of times; `gpt_light_reasoning_effort` (default `low`) rides on `light_completion` (`src/agent/openai_client.py`), the one call site the three cheap shared tasks now share. MIPRO's *prompt* model keeps the provider default — it writes a handful of candidate instructions and that is where compiled-prompt quality comes from. Empty string sends no parameter, which is what a non-reasoning model needs; `light_completion` also detects a rejection by the parameter name in the error and retries once without it, remembering the model so it isn't a per-call tax. The **token caps stayed generous on purpose** — a cap is a ceiling, not a spend, and it has to leave room for the answer if the effort is put back to default. Don't "tidy" them down.
- **A repeated question is not re-asked.** A candidate that clears the screener keeps clearing it, so the same ticker was sent to the decision model every 15 minutes — ~36 times a session — for an answer computed off *daily* bars. DSPy's own cache never catches this: the live price makes each prompt unique by a few decimals. `_pass_memo` in `scan_loop.py` reuses a stored **PASS** until it ages out (`decision_cache_minutes`, 60), the price moves `decision_recheck_move_pct` (1.5%), or the news+insider text changes. **PASS only** — a reused BUY would open a position against a stop and target placed at an older price, and BUYs are rare enough that re-asking costs nothing. The reused copy carries no `entry_inputs`, so the MIPRO/counterfactual corpus is untouched: both trainset queries filter on `entry_inputs IS NOT NULL`, and `_persist_decisions` already stored only one blob per track/ticker/day, so the day's first PASS still pays for its real call. `decision_cache_minutes=0` restores a call per scan. Likewise `analyze_news` caches on a hash of the article set (the *fetch* was already TTL-cached, so consecutive scans were buying the same summary); the jump-triggered exit review passes `use_cache=False` because freshness is what decides an exit there.
- **Per-trade P&L is net** — slippage lives in the fill prices, commissions (entry + exit legs) are subtracted in `ClosedTrade.pnl`/`pnl_pct`; pre-upgrade persisted trades default to commission 0 (gross). The equity curve, win rate, ERL and heuristic scoring all consume the net figures, and so does `rrr_achieved` (a gross R made `avg_rrr` quietly optimistic against every other number on the dashboard). `avg_rrr`/`optimization_metric` are the mean R across **all** trades (expectancy) — never average winners only. A full stop-out is therefore ≈ −1.05R, not −1.00R.
- **Entries fill at a live quote, never the scan-time OHLCV close** — the daily close a candidate was screened on can be hours stale (Alpha Vantage is EOD; EU feeds are delayed), and booking it while exits fill live realized the gap as phantom P&L (0-day ±huge trades). If no live quote exists, or it deviates more than `max_entry_price_deviation` (3%) from the scan price, the entry is blocked. News-exit fills use the same live feed. Entry-side FX resolution is *strict* — a ticker whose quote currency can't be resolved from its suffix/market (e.g. an `.IS` listing in the US watchlist) is blocked instead of booked at a guessed rate; exit pricing of existing positions stays lenient so they can still close consistently.
- **Volume is screened on the last *completed* daily bar** — intraday the latest bar is still forming, so `volume_ratio` from it reads ~0.1× and the `volume_spike_multiplier` gate would reject everything until near the close. `technical.py` computes the ratio from the previous full day vs its trailing 20-day average; `current_volume` still reports the live bar for display.


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

All model IDs are env-overridable (see `.env.example`), as is the reasoning effort each tier runs at (`GPT_DECISION_REASONING_EFFORT`, `GPT_LIGHT_REASONING_EFFORT`, `GPT_ERL_REASONING_EFFORT`). Scan/ERL models were upgraded from the original Haiku/4o-mini/Sonnet-4-6/4o tier. News analysis is a single shared GPT call (`gpt-5-mini`) fed identically to both tracks — kept on a light model, and on GPT to use the free-token quota. MIPRO uses a heavy proposer (`prompt_model`) to write candidate instructions while the cheaper decision model evaluates them.

---

## Risk rules (all enforced in `src/agent/risk.py` unless noted)

- 1% max risk per trade (hard cap 2%)
- Min RRR 2.5
- Stop-loss at 1.5× ATR below entry — validated as *fractions of price* so the check is currency-safe (entry/stop are SEK, ATR is native currency)
- **The stop is bounded from BOTH sides.** The ATR gate used to reject only stops that were too *far*, on the stated assumption that "too tight a stop is fine". It isn't: a live trade was handed a **0.26% stop**, was gone in **72 minutes**, and booked **−2.77R** — because the LSE round trip is 0.5%, so commission decided the loss rather than the move. Two floors now apply, whichever binds harder: `min_stop_atr_multiplier` (0.5×ATR, so the stop sits outside ordinary noise) and `min_stop_cost_multiple` (3× the round trip, so a stop-out is dominated by the move). Costs nothing in position size — sizing is `min(risk/stop_frac, max_position_pct)` and the value cap already binds for any stop tighter than 10%. `min_atr_pct` does not cover this: it screens the *candidate's* volatility, not the *model's* stop placement
- **`max_position_pct` (10%) is the throughput knob, not just a funding guard** — concurrent positions ≈ `market_allocation` ÷ position size, so at 25% the whole book held *two* positions and a 25% position could not fit the 20% EU budget at all (EU was structurally untradeable, unlogged). At ~9-day holds that was 135 days to the 30 trades MIPRO needs; 10% gives 10 slots and ~27. The cost is that the value cap now binds nearly always, so actual risk lands at ~0.3–0.9% of equity and `max_risk_per_trade` rarely applies — but R is invariant to position size, so the learning signal is untouched. Don't raise it back for a livelier equity curve; that trades learning rate for cosmetics.
- >10% portfolio drawdown → halve all position sizes
- No duplicate tickers across open positions; max 2 positions per sector
- Pairwise return-correlation cap: candidate vs each same-market open position (60-day daily returns from the scan's batch OHLCV); any pair > `max_sector_correlation` (0.7) rejects the entry — same rule in the backtester
- **Exits are labelled by the level that actually bound.** The trail (`trailing_stop_atr_multiplier`, 2×ATR) is wider than the maximum permitted stop (1.65×ATR), so `peak − trail` sits *below* entry until price has run a full 2 ATR. Labelling a trailed exit by "did the trail ratchet above stop_loss" therefore called ≈−1R losses `trailing_stop` — it did, on 24 of the first 56 trades. An independent breakeven floor arms at `breakeven_arm_atr_multiplier` (1×) ATR of profit, is computed net of both commission legs and exit slippage, and never retreats. `resolve_exit()` labels `trailing_stop` only when the trail locked a gain **above entry**, `breakeven_stop` when the armed floor caught the reversal, `stop_loss` otherwise. ERL branches on this, so the old labels had it blaming trade management for entry-selection failures.
- **A winner pays `risk × RRR`, not the size of the move** — sizing is risk-normalised, so payoff is independent of ATR and how far the target sits. Both levers are therefore load-bearing: the `TradeDecision` docstring places stop/target from *structure* and carries the numeric floor on the output-field description (stating it in the docstring anchored every target at the minimum), and the screener enforces `min_atr_pct` plus a range score term so candidates can actually travel. Don't reintroduce a floor-shaped worked example.
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
src/analysis/regime.py      Hurst Exponent + autocorrelation → trending/mean-reverting
src/analysis/screener.py    Multi-factor filter → top-N candidates
src/agent/decision.py       DSPy TradeDecision program; DecisionEngine per track
src/agent/risk.py           Position sizing, stop validation, RRR check
src/agent/memory.py         HeuristicStore — file-backed, track-namespaced
src/agent/erl.py            Post-trade causal analysis → heuristic extraction
src/agent/news_analyzer.py  Shared per-ticker news analysis (gpt-5-mini, both tracks); caches on the article set
src/agent/openai_client.py  light_completion() — the one call site for triage/news/watch; applies reasoning_effort
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
```

---

## Learning loop (how the system improves)

- **MIPRO scores R-multiples, not raw returns.** Position size is already risk-normalised, so a metric denominated in percent rewards volatility for its own sake; and the old `tanh(pnl × 10)` saturated so hard that a +30% trade scored 0.045 above a +15% one, which would have silently undone the re-tune for a fatter right tail. `_R_METRIC_SCALE = 0.35` keeps 2.5R and 5R 0.12 apart. A BUY earns its R and **a PASS earns the R it avoided** — so there is no fixed 0.5 baseline. PASS used to score a flat 0.5, which meant no credit for dodging a loser: measured on a real 395-example corpus, an oracle with 100% precision *and* recall beat do-nothing by **0.052**, because 81% of setups don't pay and 0.5 was the ceiling on those. Crediting the avoided loss doubles the band to 0.104. It does *not* make always-pass lose — on a corpus with no edge it still beats always-buy, correctly; what changes is that the oracle pulls clearly ahead of both, which is what selection needs. Read `headroom` (oracle − best trivial) from the replay harness: near zero means no prompt is distinguishable on that corpus, however many examples it holds.
- **MIPRO trainset** = real closed trades (labeled by realized P&L) **plus counterfactuals**: PASS decisions persist their DSPy inputs + decision-time price (one blob per track/ticker/day); at MIPRO time they're labeled from the forward return over `counterfactual_horizon_days` (≥3% → missed BUY, ≤0 → correct PASS, middle skipped). Counterfactual rows are diversity-sampled by `select_decision_rows` (frequency-ordered, ≤5 per ticker) — taking the most recent N let a few frequently-decided tickers supply most of the half. Counterfactuals are capped at `counterfactual_ratio_cap` (4×) the real-trade count, not at parity — parity tied the trainset to the scarcest input and left MIPRO selecting instructions on a 12-example validation split.
- **Heuristic feedback runs on both sides of a decision.** `heuristic_ids` ride in `entry_inputs` (added in `scan_loop`, *never* passed into the DSPy program call) on opened positions **and on PASS/BLOCKED decisions**. On close, `record_outcome` moves quality by up to ±1, rank-weighted, clamped 0–10. Weekly, `score_heuristics_from_decisions` scores aged skipped setups from their counterfactual forward label — sign follows the decision the rules informed, so a PASS is right when the path went nowhere while a BLOCKED BUY scores like a taken trade. Scoring only opened trades judged a rule on the subset of its influence that produced a position, which is the same survivorship bias the counterfactual trainset exists to remove. Idempotent via `decisions.heuristics_scored`; `record_outcome` has none of its own and would double-count.
- **Core status is earned and revocable** — `promote_core` needs access count **and** quality; a core rule whose quality decays loses the flag. Access count only measures how often a rule was *shown*, so promoting on it alone handed a permanent +2.0 retrieval boost to whatever matched the commonest regime, which then made it show up more. New rules are extracted from a single trade, so they carry a retrieval penalty until `MIN_CORROBORATION` outcomes — ranked down, never excluded, or they could never earn the outcomes that prove them.
- **News prefilter** matches the company name from `universe.csv` (headlines say "Volvo", never "VOLV-B"). Universe names are legal names ("Telefonaktiebolaget LM Ericsson (publ)"), so the matcher strips parentheticals + corporate-form words and takes the last distinctive token ("ericsson").

---

## What's left to build

See [STATUS.md](STATUS.md) for the full To Do list. Priority items:

1. **Flip `hurst_on_returns`** — the returns-based R/S estimator is implemented behind a settings flag (default off, because it reclassifies drifting walks as neutral and makes the screener stricter); enable deliberately and observe candidate volume
2. **News summary quality** — monitor whether `gpt-5-mini` spends its budget on reasoning at the expense of the Swedish summaries

**A prompt can be evaluated offline.** `src/agent/replay.py` + `scripts/replay_decisions.py` label aged PASS/BLOCKED decisions from what the price then did (`_label_forward_path`, so identical ground truth to the MIPRO trainset) and replay any program over them with `program(**entry_inputs)` — the same call the live path makes. The backtester cannot do this: it contains no model and buys every screener survivor. Build the corpus once (network-bound, cached to JSON), then score programs against the cache. `--reference` scores oracle / always-buy / always-pass with **no LLM calls** — if those three don't order correctly the harness is broken, so run it first. `decision_metric` lives in `metrics.py` rather than beside MIPRO precisely so the harness can score without importing dspy.

There is **no target auto-stretching**: a BUY whose own target gives RRR < `min_rrr` is rejected at risk validation and learned from counterfactually (blocked BUYs persist inputs like PASSes). Don't reintroduce `_fix_rrr`.

The backtester now mirrors live execution (slippage/commissions, intraday High/Low exits, ATR trailing stop, mark-to-market equity, correlation cap); counterfactual labels simulate the stop/target path **and the breakeven floor** when ATR is available — without that mirror, hindsight BUY examples carried a clean +min_rrr while lived trades of the same setup carried a floored result, and the shared P&L metric rewarded BUY on the counterfactual half for free.

---

## Style conventions

- No comments unless the WHY is non-obvious
- No docstrings longer than one line
- Trust imports — don't add redundant `try/except` around internal calls
- Type hints on all function signatures
- `from __future__ import annotations` at top of every file
