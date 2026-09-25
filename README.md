# DeepSwing

AI-powered swing trading simulator running on a Raspberry Pi 5. Paper-trades Nordic, EU and US markets using two parallel AI simulation tracks — one powered by Claude, one by GPT — so their decision quality can be compared over time. Prompt candidates are proposed by a bounded DSPy search, screened historically, and kept inactive for prospective evaluation.

**No real money is involved. Simulation only.**

---

## What it does

- Scans three markets on their own calendars: **Nordic** (OMXS30 + the wider `universe.csv`, 09:00–17:30 CET), **EU** (09:00–17:30 CET) and **US** (09:30–16:00 ET — evaluated in Eastern Time, since US and EU switch DST weeks apart)
- Computes 11 technical indicators (EMA, SMA, ATR, Bollinger Bands, RSI, Parabolic SAR, OBV, EOM, Fibonacci, volume ratio)
- Classifies market regime via Hurst Exponent (trending vs. mean-reverting → different entry tactics)
- Screens the universe, then a **cheap shared triage call** ranks the survivors so only the top few reach the expensive models
- Pulls news via NewsAPI + Swedish RSS (yfinance/Finnhub fallback for US) → a shared `gpt-5-mini` call analyses per-ticker relevance and sentiment for both tracks
- Incorporates macro context (FRED, Riksbank, ECB) and insider activity (SEC EDGAR, FI Insynsregistret)
- Two AI tracks independently return **BUY / PASS** on candidates and **HOLD / SELL** on holdings, via DSPy-structured prompts
- Risk engine enforces 1% risk per trade, a 10% position-value cap, minimum 2.5 RRR, and ATR-based stops bounded from **both** sides (too tight is as dangerous as too wide — a stop inside daily noise is decided by commission, not by the move)
- Stops and targets are swept on their own **15-minute timer**, independent of the 30-minute scan, because an exit fills at the price it is observed at
- After each closed trade, Experiential Reflective Learning (ERL) extracts a reusable trigger→action heuristic
- Weekly bounded search proposes one prompt per track from closed trades **plus counterfactually-labelled PASS decisions**, then screens it without activating it
- Personal watchlist alerts to Telegram on large day moves, fresh directional news and insider activity
- Web dashboard with head-to-head comparison of both tracks, heuristic library, trade history

---

## Models

| Task | Claude track | GPT track |
|---|---|---|
| Scan decisions (30-min) | `claude-sonnet-5` | `gpt-5` |
| ERL causal analysis | `claude-opus-4-8` + adaptive thinking | `gpt-5.6-sol`, `reasoning_effort=high` |
| News analysis | `gpt-5-mini` (shared by both tracks) | |
| Candidate triage | `gpt-5-mini` (shared by both tracks) | |
| Prompt proposer / task model | `claude-opus-4-8` / `claude-sonnet-5` | `gpt-5.6-sol` / `gpt-5` |

All model IDs and reasoning-effort tiers are env-overridable — see `.env.example`. Reasoning tokens bill as output tokens, so `GPT_DECISION_REASONING_EFFORT` and `GPT_LIGHT_REASONING_EFFORT` are the two knobs that most affect the OpenAI bill.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM clients | `anthropic`, `openai` |
| Prompt optimization | `dspy-ai` — one bounded instruction proposal + paired screen (weekly, Sunday 02:00 CET); MIPROv2 is explicit legacy mode |
| Technical analysis | `ta` (pure Python, Pi-safe) |
| Market calendars | `exchange_calendars` (XSTO / XETR / XNYS — holidays and half-days) |
| Data | `yfinance` (US + Nordic fallback), `alpha_vantage` (Nordic primary), shared `financedata` library |
| Database | SQLite via `sqlalchemy` |
| Web | `fastapi` + `uvicorn` + WebSocket + Chart.js |
| Scheduler | `apscheduler` — 30-min market-hours-aware scan, independent 15-min stop sweep, 15-min watchlist monitor |
| Deployment | Raspberry Pi 5, systemd service, Cloudflare Tunnel for custom domain |

---

## Quick Start

```bash
git clone https://github.com/ACWesterberg/DeepSwing.git
cd DeepSwing
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
venv/bin/python main.py
```

Dashboard: `http://localhost:8000`

Trigger a scan without waiting for the scheduler:

```bash
curl -X POST http://localhost:8000/api/scan/nordic
```

See [SETUP.md](SETUP.md) for full Raspberry Pi deployment and custom domain (Cloudflare Tunnel) instructions.

---

## API Keys

| Key | Where to get | Free tier | Required? |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com | Pay-per-use | yes — Claude track |
| `OPENAI_API_KEY` | platform.openai.com | Pay-per-use | yes — GPT track, plus shared news/triage |
| `ALPHA_VANTAGE_API_KEY` | alphavantage.co | 25 req/day | Nordic prices (yfinance fallback) |
| `NEWS_API_KEY` | newsapi.org | 100 req/day | optional — RSS + yfinance back it up |
| `FRED_API_KEY` | fred.stlouisfed.org | Free | macro context |
| `FINNHUB_API_KEY` | finnhub.io | 60 req/min | optional — preferred US per-ticker news |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | @BotFather | Free | optional — watchlist alerts stay dormant without them |

---

## Documentation

Run tests with `python -m pip install -r requirements-dev.txt` followed by
`python -m pytest -q`. Market/provider calls are mocked in the unit suite; the
DSPy integration test uses a local dummy model in a separate process with
network connections blocked.

The [improvement progress](reviews/implementation-progress.md) documents the
2026-09-23 correctness and prompt-governance improvements, including the required
rebuild of old replay caches. Optimizer candidates must pass a fresh temporal
test and are then registered inactive under `compiled/candidates/`; scheduled
optimization never replaces the active prompt. Activation is a separate operation
that requires recorded forward-evidence approval. Reports and snapshots are saved under
`compiled/evaluations/`. Scheduled optimization evaluates predicted stops and targets on frozen
daily price paths. Use replay `build --plans` and `score --plans` for the same
plan evaluator; the default replay mode retains the older action-only score.
Plan scores are isolated opportunity results. Offline `score --portfolio
--track claude` replays the sampled opportunities with shared cash, risk sizing,
allocation limits, correlation checks and dated NAV. It assumes constant FX and
reports missing coverage; see the progress document for commands and limits.

Prospective shadow evaluation is disabled by default (`SHADOW_ENABLED=false`).
When enabled, at most the configured number of paid candidate requests is
reserved per day. Each case freezes the incumbent inputs, both outputs, entry
quote, ATR and execution policy; candidate outputs never reach execution.
Failures and ambiguous started requests are retained, exact requests are reused,
and another paid attempt requires explicit retry authorization. A daily no-LLM
collector freezes mature forward paths and scores chronologically selected,
non-overlapping 14-day windows with equal period weight. Overlapping cases remain
stored and are reported as excluded from gate scoring.

Shadow cases freeze expected trading sessions using the listing's exchange calendar. Missing sessions, duplicate bars, or bars on non-session dates prevent outcome completion. Historical plan corpora use the same check. This checks calendar coverage; it does not validate corporate actions, trading halts, or vendor price accuracy. Existing plan corpora must be rebuilt after execution-code changes.

Scheduled prompt search defaults to `PROMPT_SEARCH_MODE=bounded`: one proposed
instruction is compared with the incumbent on eight chronologically spread
held-out opportunities covering at least six tickers and four decision days.
The worst case is 17 paid calls (one proposal plus two arms); exact completed
requests are reused. Request, input-byte and cumulative output reservations are
written before calls. Failed or ambiguous requests stop the run and cannot be
silently resubmitted. This is an inexpensive historical screen, not evidence of
an investment advantage; passing candidates remain inactive for forward testing.

Provider-returned token usage for durable bounded-search attempts can be inspected without making API calls:

```bash
python scripts/report_optimizer_usage.py
```

The report keeps missing usage explicit and treats cache/reasoning counters as subsets of provider input/output totals, not extra tokens to add again. Its advisory dollar result uses a dated exact-model allowlist and becomes unavailable rather than presenting stale, unsupported, or partial pricing.

The default report includes synchronous and Batch attempts, with pricing applied per attempt according to its transport. Uncertain Batch submissions count as unknown usage; shared request IDs within a batch count once. `--root` selects a compiled artifact directory; use `--cache-only --root <search-cache-directory>` for the older cache-only view. Raw prompts and response bodies are omitted from the report.

The pricing snapshot supports GPT-5 (including its verified dated snapshot) and GPT-5.6 Sol. Sol estimates require cache-read/write counters and apply long-context rates per request; missing counters produce an unavailable quote. Anthropic and other unverified models remain unpriced. Quotes assume standard processing or the recorded Batch transport and exclude account-specific charges and taxes.

An hourly job retrieves existing OpenAI Batch checkpoints and scores complete results using frozen thresholds and artifacts. It stays quiet while a batch is pending and notifies when scoring finishes or intervention is required; failed notification delivery is retried. New optimizer Batch submissions require `BOUNDED_SEARCH_OPENAI_BATCH=true` and remain disabled by default. The collector cannot initiate paid requests. Passing candidates remain inactive pending forward evaluation.

Failed Batch requests can be retried explicitly with `python scripts/retry_optimizer_batch.py <run-directory> --authorized-by <name> --reason <reason> --submit`. This is a paid operation. It retries provider failures and unparseable completions, preserves successes, and checks the original run's cumulative budget. Exhausted budgets require explicit new total ceilings through `--max-requests` and `--max-reserved-output-tokens`; previous consumption is retained. `--output-tokens` can raise the allowance for failed requests, but the changed comparison then ends in `historical_review_required`, without candidate registration. Retry checkpoints and authorization records live under the run's `retries/` directory. The normal collector retrieves submitted retries automatically. Uncertain submissions require explicit batch-ID adoption using `adopt_batch_id`, never a fresh submission.

| Doc | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here.** Purpose, every design decision and the failure that motivated it, models, risk rules, file map, learning loop |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System flow and module layout |
| [STATUS.md](STATUS.md) | What's built, what's next |
| [SETUP.md](SETUP.md) | Raspberry Pi provisioning, Cloudflare Tunnel, network watchdog |

Sibling project on the same Pi: **ai-fund-manager** — a weekly LLM portfolio allocator (one book holds real money). Different problem, shared `financedata` library.
