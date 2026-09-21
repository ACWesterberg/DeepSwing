# DeepSwing

AI-powered swing trading simulator running on a Raspberry Pi 5. Paper-trades Nordic, EU and US markets using two parallel AI simulation tracks — one powered by Claude, one by GPT — so their decision quality can be compared over time. Prompts evolve automatically via DSPy/MIPRO optimization using closed trades as training data.

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
- Weekly MIPRO optimization compiles improved prompts per track, trained on closed trades **plus counterfactually-labelled PASS decisions** — so the model learns from setups it declined, not only from the ones it took
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
| MIPRO proposer / task model | `claude-opus-4-8` / `claude-sonnet-5` | `gpt-5.6-sol` / `gpt-5` |

All model IDs and reasoning-effort tiers are env-overridable — see `.env.example`. Reasoning tokens bill as output tokens, so `GPT_DECISION_REASONING_EFFORT` and `GPT_LIGHT_REASONING_EFFORT` are the two knobs that most affect the OpenAI bill.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| LLM clients | `anthropic`, `openai` |
| Prompt optimization | `dspy-ai` — DSPy `TradeDecision` signature + MIPROv2 (weekly, Sunday 02:00 CET) |
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
2026-09-21 correctness and prompt-promotion improvements, including the required
rebuild of old replay caches. Optimizer candidates must pass a fresh temporal
test before replacing an active prompt; reports and snapshots are saved under
`compiled/evaluations/`. Scheduled optimization now evaluates predicted stops and targets on frozen
daily price paths. Use replay `build --plans` and `score --plans` for the same
plan evaluator; the default replay mode retains the older action-only score.
Plan scores are isolated opportunity results. Offline `score --portfolio
--track claude` replays the sampled opportunities with shared cash, risk sizing,
allocation limits, correlation checks and dated NAV. It assumes constant FX and
reports missing coverage; see the progress document for commands and limits.

| Doc | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here.** Purpose, every design decision and the failure that motivated it, models, risk rules, file map, learning loop |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System flow and module layout |
| [STATUS.md](STATUS.md) | What's built, what's next |
| [SETUP.md](SETUP.md) | Raspberry Pi provisioning, Cloudflare Tunnel, network watchdog |

Sibling project on the same Pi: **ai-fund-manager** — a weekly LLM portfolio allocator (one book holds real money). Different problem, shared `financedata` library.
