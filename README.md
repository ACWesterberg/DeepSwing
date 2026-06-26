# DeepSwing

AI-powered swing trading simulator running on a Raspberry Pi 5. Paper-trades Nordic (OMXS30) and US (S&P 500) markets using two parallel AI simulation tracks — one powered by Claude, one by GPT — so their decision quality can be compared over time. Prompts evolve automatically via DSPy/MIPRO optimization using closed trades as training data.

**No real money is involved. Simulation only.**

---

## What it does

- Scans 30 Nordic stocks during Stockholm session (09:00–17:30 CET) and top-100 US stocks during Wall Street session (15:30–22:00 CET)
- Computes 11 technical indicators (EMA, SMA, ATR, Bollinger Bands, RSI, Parabolic SAR, OBV, EOM, Fibonacci)
- Classifies market regime via Hurst Exponent (trending vs. mean-reverting → different entry tactics)
- Pulls news via NewsAPI + Swedish RSS feeds → Claude Haiku analyzes per-ticker relevance and sentiment
- Incorporates macro context (FRED, Riksbank, ECB) and insider activity (SEC EDGAR, FI Insynsregistret)
- Two AI tracks make independent BUY/SELL/HOLD decisions using DSPy-structured prompts
- Risk engine enforces 1% position sizing, minimum 2.0 RRR, ATR-based stop-losses
- After each closed trade, runs Experiential Reflective Learning (ERL) — causal analysis extracts reusable heuristics stored as JSON rules
- Weekly MIPRO optimization compiles improved prompts for each track independently
- Web dashboard with head-to-head comparison of both tracks, heuristic library, trade history

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Claude track | `anthropic` — Haiku 4.5 (decisions), Sonnet 4.6 + extended thinking (ERL) |
| GPT track | `openai` — GPT-4o-mini (decisions), GPT-4o (ERL) |
| Prompt optimization | `dspy-ai` — DSPy `TradeDecision` signature + MIPROv2 |
| Technical analysis | `ta` (pure Python, Pi-safe) |
| Data | `yfinance` (US + Nordic fallback), `alpha_vantage` (Nordic primary) |
| Database | SQLite via `sqlalchemy` |
| Web | `fastapi` + `uvicorn` + Chart.js |
| Scheduler | `apscheduler` (market-hours-aware, 15 min scan interval) |
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

See [SETUP.md](SETUP.md) for full Raspberry Pi deployment and custom domain (Cloudflare Tunnel) instructions.

---

## API Keys Required

| Key | Where to get | Free tier |
|---|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com | Pay-per-use |
| `OPENAI_API_KEY` | platform.openai.com | Pay-per-use |
| `ALPHA_VANTAGE_API_KEY` | alphavantage.co | 25 req/day |
| `NEWS_API_KEY` | newsapi.org | 100 req/day |
| `FRED_API_KEY` | fred.stlouisfed.org | Free |

---

## Project Status

See [STATUS.md](STATUS.md) for current implementation progress and what's planned next.
See [ARCHITECTURE.md](ARCHITECTURE.md) for full system design and data flow.
