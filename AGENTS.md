# AGENTS.md

## Cursor Cloud specific instructions

DeepSwing is a single Python process (`main.py`) that runs a FastAPI + uvicorn dashboard,
an APScheduler background scanner, and an embedded SQLite DB (`data/deepswing.db`, auto-created).
Dashboard serves on port **8000**. See `CLAUDE.md` / `README.md` / `SETUP.md` for architecture and standard commands.

### Environment layout (already provisioned by the update script)
- Python deps live in a venv at `/workspace/venv`. Always invoke via `venv/bin/python` / `venv/bin/pip`.
- **`financedata` is an external dependency that is NOT in this repo and NOT on PyPI.** It is the
  sibling repo `ACWesterberg/FinanceData`, cloned to `$HOME/FinanceData` and installed editable.
  The app cannot even import without it (`main.py` → `scan_loop` → `market_data` → `from financedata import ...`).
  If imports of `financedata` fail, re-run the update script.
- `pytest` is a dev-only tool (not pinned in `requirements.txt`); the update script installs it into the venv.

### Running / testing / building
- Run the app: `venv/bin/python main.py` (dashboard at http://localhost:8000). There is no separate build step.
- Tests: `venv/bin/python -m pytest -q` (206 tests; `tests/conftest.py` stubs `financedata`, `dspy`, `anthropic`, `openai`, etc., so tests do not need real keys or network).
- No linter is configured in this repo (no ruff/flake8/black config); pytest is the automated check.

### API keys / boot behavior (non-obvious)
- The app **boots and serves the dashboard even with no API keys**; unconfigured LLM providers are logged and skipped by the startup preflight (it never raises).
- `OPENAI_API_KEY` is available in this cloud environment as an injected secret, so the **GPT track is fully functional** (real scan decisions, news analysis). `ANTHROPIC_API_KEY` is **not** provided, so the **Claude track's decision/ERL calls fail with 401** — this is expected, not a bug.
- `.env` is created from `.env.example` (gitignored). Its placeholder values (e.g. `NEWS_API_KEY=your_..._here`) cause harmless 401s; per-ticker news then falls back to a free source (yfinance/Yahoo), which works. Real env-var secrets override `.env` values.

### Exercising the product
- Manual scan (does not wait for the 15-min scheduler, and bypasses market-hours gating):
  `curl -X POST http://localhost:8000/api/scan/us` (or `/nordic`). A scan runs fetch → screen → news → per-track LLM decision → risk-sized paper trade. With the GPT track live, this opens real paper positions visible under the "GPT Track" tab and `/api/portfolio/gpt`.
- When `ANTHROPIC_API_KEY` is missing, each candidate's Claude decision retries before failing, so scans are noticeably slower than with both keys set.
- The `/api/backtest` walk-forward endpoint defaults to the **full Nordic universe (600+ tickers)** and is very slow (many minutes, steps day-by-day). Avoid it for quick checks, or scope it down.
