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
- **Capacity-aware scanning** — a track with free cash below `min_cash_for_new_position_pct` (5%) of its equity is treated as fully allocated and gets no entry decisions; when *no* track is funded the scan skips the whole candidate/news/decision pipeline and runs a holdings-only monitor. Holdings are tracked on price alone — a news pull + AI exit review only fires once a position moves ≥ `holdings_news_jump_pct` (5%) since its last check (closes as `exit_reason="news_exit"`). Set either knob to 0 to restore always-on behaviour.
- **Portfolio state is durable** — the live `Portfolio` (cash, open positions, closed trades, peak equity) is an in-memory object mirrored to the `portfolio_state` DB table on every open/close and at end of scan, and rehydrated on startup (`persistence.restore_portfolios()`), so tracks survive a redeploy/restart. `main.py` restores *before* arming the persistence handler; `/api/reset` deletes the persisted rows so a restart doesn't resurrect cleared tracks. Heuristics stay file-backed; MIPRO programs stay git-backed.
- **Scans never block the event loop** — `run_scan` is long/blocking (network + LLM), so `/api/scan` offloads it via `run_in_executor`; a module-level `_scan_lock` serializes scans so a manual trigger can't overlap the scheduled one and double-open. The scheduler already runs scans in its own thread.
- **NewsAPI is rate-limit-guarded** — per-ticker news is cached for `news_refresh_interval_minutes`; if a fetch stalls beyond `newsapi_slow_threshold_seconds` (429 backoff) a breaker skips NewsAPI (RSS only) for `newsapi_cooldown_minutes`, so one throttled ticker doesn't cost ~1 min each. The jump-triggered exit review passes `force_refresh=True` for freshness.

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
| ERL causal analysis | `claude-opus-4-8` + extended thinking | `gpt-5.5` + `reasoning_effort=high` |
| News analysis | `gpt-5-mini` (shared by both tracks) | `gpt-5-mini` |
| MIPRO — task model (evaluates candidates) | `claude-sonnet-5` | `gpt-5` |
| MIPRO — prompt model (writes instructions) | `claude-opus-4-8` | `gpt-5.5` |

All model IDs are env-overridable (see `.env.example`). Scan/ERL models were upgraded from the original Haiku/4o-mini/Sonnet-4-6/4o tier. News analysis is a single shared GPT call (`gpt-5-mini`) fed identically to both tracks — kept on a light model, and on GPT to use the free-token quota. MIPRO uses a heavy proposer (`prompt_model`) to write candidate instructions while the cheaper decision model evaluates them.

---

## Risk rules (all enforced in `src/agent/risk.py`)

- 1% max risk per trade (hard cap 2%)
- Min RRR 2.0
- Stop-loss at 1.5× ATR below entry
- >10% portfolio drawdown → halve all position sizes
- No duplicate tickers across open positions

---

## File map (key files)

```
config/settings.py          All config — API keys, risk params, model names, watchlists
src/db.py                   SQLAlchemy models (Trade, Position, PortfolioSnapshot, PortfolioState, Heuristic, Decision)
src/portfolio/persistence.py  DB save/restore of live portfolio state (survives restarts)
src/data/market_data.py     OHLCV fetch — yfinance + Alpha Vantage
src/data/news_fetcher.py    NewsAPI + Swedish RSS
src/data/insider_fetcher.py SEC EDGAR + FI Insynsregistret
src/data/macro_data.py      FRED + Riksbank + ECB
src/analysis/technical.py   11 indicators via `ta` library
src/analysis/regime.py      Hurst Exponent + autocorrelation → trending/mean-reverting
src/analysis/screener.py    Multi-factor filter → top-N candidates
src/agent/decision.py       DSPy TradeDecision program; DecisionEngine per track
src/agent/risk.py           Position sizing, stop validation, RRR check
src/agent/memory.py         HeuristicStore — file-backed, track-namespaced
src/agent/erl.py            Post-trade causal analysis → heuristic extraction
src/agent/news_analyzer.py  Claude Haiku per-ticker news analysis
src/portfolio/simulator.py  Paper trading engine (Portfolio class); dual-track
src/portfolio/metrics.py    Sharpe, drawdown, win rate, MIPRO metric
src/scheduler/market_hours.py  is_market_open(), active_markets(), CET-aware
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

## What's left to build

See [STATUS.md](STATUS.md) for the full To Do list. Priority items:

1. **Fix ERL input capture** — `scan_loop.py` passes placeholder technicals string to ERL; needs to store the real technical snapshot at trade entry on the Position object
2. **Fix drawdown mode wiring** — `risk.py::_is_drawdown_mode()` returns `False` (placeholder); `scan_loop.py` needs to pass `portfolio.is_drawdown_mode` into the risk validator
3. **Sector correlation enforcement** — currently only blocks duplicate tickers; full sector correlation matrix needed
4. **Unit tests** — `tests/` is empty; priority: technical.py, regime.py, risk.py, screener.py
5. **Pi deployment** — deploy, verify scheduler, set up Cloudflare Tunnel

---

## Style conventions

- No comments unless the WHY is non-obvious
- No docstrings longer than one line
- Trust imports — don't add redundant `try/except` around internal calls
- Type hints on all function signatures
- `from __future__ import annotations` at top of every file
