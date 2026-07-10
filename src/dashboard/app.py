from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

_STATIC_VERSION = str(int(time.time()))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from config.settings import settings
from src.portfolio.metrics import build_equity_curve_chart_data, compute_metrics
from src.portfolio.simulator import get_portfolio, reset_portfolios
from src.scheduler.market_hours import active_markets, is_exchange_open
from src.scheduler.markets import SCAN_MARKETS
from src.scheduler.scan_loop import clear_recent_decisions, get_recent_decisions, run_scan, set_trade_event_handler

logger = logging.getLogger(__name__)

app = FastAPI(title="DeepSwing Dashboard", version="1.0.0")


_SESSION_COOKIE = "ds_session"
_SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days

# Server-side session store: random token → expiry epoch. The cookie must never
# carry the password itself — a leaked cookie would leak the credential and be
# irrevocable. In-memory, so a restart logs everyone out (acceptable).
_sessions: dict[str, float] = {}


def _token_valid(token: str) -> bool:
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if expiry < time.time():
        _sessions.pop(token, None)
        return False
    return True


def _valid_session(request: Request) -> bool:
    """Return True if the request carries a valid session cookie."""
    return _token_valid(request.cookies.get(_SESSION_COOKIE, ""))


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.dashboard_password:
            return await call_next(request)

        path = request.url.path

        # Always allow static assets and the login form itself
        if path.startswith("/static/") or path == "/login":
            return await call_next(request)

        # Valid session cookie — let everything through
        if _valid_session(request):
            return await call_next(request)

        # API calls and WebSocket get 401 (not a redirect — JS can't follow login redirects)
        if path.startswith("/api/") or path == "/ws":
            return Response("Unauthorized", status_code=401)

        # Browser navigation — redirect to login page
        return Response(
            status_code=302,
            headers={"Location": "/login"},
        )

app.add_middleware(_AuthMiddleware)

_template_dir = str(__file__).replace("app.py", "templates")
_static_dir = str(__file__).replace("app.py", "static")

templates = Jinja2Templates(directory=_template_dir)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Active WebSocket connections
_ws_clients: list[WebSocket] = []


@app.on_event("startup")
async def _register_trade_event_handler() -> None:
    """Wire scan_loop trade events into the WebSocket broadcast."""
    loop = asyncio.get_event_loop()

    def _sync_emit(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(_broadcast(event), loop)

    set_trade_event_handler(_sync_emit)


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>DeepSwing — Login</title>
<style>
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
form{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:32px;display:flex;flex-direction:column;gap:14px;width:280px}
h2{margin:0;font-size:18px;text-align:center}
input{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px 10px;color:#e6edf3;font-size:14px;outline:none}
input:focus{border-color:#7c3aed}
button{background:#7c3aed;border:none;border-radius:6px;padding:9px;color:#fff;font-size:14px;cursor:pointer}
button:hover{background:#6d28d9}
.err{color:#f85149;font-size:13px;text-align:center;display:none}
</style></head>
<body><form method="POST" action="/login">
<h2>DeepSwing</h2>
<input name="password" type="password" placeholder="Password" autofocus autocomplete="current-password"/>
<button type="submit">Sign in</button>
<div class="err" id="e"></div>
</form></body></html>""")


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    pwd = form.get("password", "")
    if settings.dashboard_password and secrets.compare_digest(str(pwd), settings.dashboard_password):
        token = secrets.token_urlsafe(32)
        _sessions[token] = time.time() + _SESSION_TTL_SECONDS
        resp = Response(status_code=302, headers={"Location": "/"})
        resp.set_cookie(
            _SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=_SESSION_TTL_SECONDS,
        )
        return resp
    return Response(status_code=302, headers={"Location": "/login?err=1"})


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html", {"v": _STATIC_VERSION})


@app.get("/api/status")
async def status():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_markets": active_markets(),
        "nordic_open": is_exchange_open("nordic"),
        "eu_open": is_exchange_open("eu"),
        "us_open": is_exchange_open("us"),
        "tracks": settings.tracks,
        "claude_configured": bool(settings.anthropic_api_key),
        "gpt_configured": bool(settings.openai_api_key),
    }


@app.get("/api/portfolio/{track}")
async def portfolio_state(track: str):
    if track not in settings.tracks:
        return {"error": f"Unknown track: {track}"}
    portfolio = get_portfolio(track)
    metrics = compute_metrics(portfolio)
    return {
        "snapshot": portfolio.snapshot(),
        "metrics": metrics.to_dict(),
        "open_positions": [p.to_dict() for p in portfolio.open_positions],
    }


@app.get("/api/trades/{track}")
async def trade_history(track: str, limit: int = 50):
    if track not in settings.tracks:
        return {"error": f"Unknown track: {track}"}
    portfolio = get_portfolio(track)
    trades = sorted(portfolio.closed_trades, key=lambda t: t.exit_time, reverse=True)[:limit]
    return {"trades": [t.to_dict() for t in trades]}


@app.get("/api/comparison")
async def comparison():
    """Side-by-side metrics for both tracks."""
    result = {}
    for track in settings.tracks:
        portfolio = get_portfolio(track)
        metrics = compute_metrics(portfolio)
        result[track] = {
            "metrics": metrics.to_dict(),
            "snapshot": portfolio.snapshot(),
            "equity_curve": build_equity_curve_chart_data(portfolio),
        }
    return result


@app.get("/api/heuristics/{track}")
async def heuristics(track: str, page: int = 1, page_size: int = 20):
    if track not in settings.tracks:
        return {"error": f"Unknown track: {track}"}
    from src.agent.memory import get_store
    store = get_store(track)
    all_h = sorted(
        store.all_as_list(),
        key=lambda h: h.get("quality_score", 0) * max(h.get("access_count", 1), 1),
        reverse=True,
    )
    start = (page - 1) * page_size
    return {
        "total": len(all_h),
        "page": page,
        "heuristics": all_h[start: start + page_size],
    }


@app.get("/api/decisions")
async def decisions():
    """Latest per-market scan decisions (action + reasoning) for all tracks."""
    return get_recent_decisions()


@app.get("/api/decisions/history")
async def decisions_history(
    limit: int = 100,
    track: str = "",
    action: str = "",
    ticker: str = "",
):
    """Persisted decision history, newest first, with optional filters."""
    from sqlalchemy import desc
    from src.db import Decision, get_session

    session = get_session()
    try:
        query = session.query(Decision)
        if track:
            query = query.filter(Decision.track == track)
        if action:
            query = query.filter(Decision.action == action.upper())
        if ticker:
            query = query.filter(Decision.ticker.ilike(f"%{ticker}%"))
        rows = (
            query.order_by(desc(Decision.timestamp))
            .limit(max(1, min(limit, 500)))
            .all()
        )
        return {"decisions": [r.to_dict() for r in rows]}
    finally:
        session.close()


@app.post("/api/backtest")
async def run_backtest(
    market: str = "us",
    start: str = "",
    end: str = "",
    n_windows: int = 4,
    initial_equity: float = 100_000.0,
):
    """
    Run walk-forward backtesting on historical data.
    No AI calls — uses ATR-based entries, real screener + risk rules.
    """
    if market not in SCAN_MARKETS:
        return {"error": f"market must be one of {SCAN_MARKETS}"}

    from src.backtesting.engine import BacktestEngine
    from src.data.watchlist import get_eu_watchlist, get_omxs30_tickers, get_us_tickers

    try:
        start_date = date.fromisoformat(start) if start else date(date.today().year - 1, 1, 1)
        end_date = date.fromisoformat(end) if end else date.today()
    except ValueError as exc:
        return {"error": f"Invalid date format: {exc}"}

    if market == "nordic":
        tickers = get_omxs30_tickers()
    elif market == "eu":
        tickers = get_eu_watchlist()
    else:
        tickers = get_us_tickers()

    import asyncio
    loop = asyncio.get_event_loop()
    engine = BacktestEngine(
        market=market,
        tickers=tickers,
        start=start_date,
        end=end_date,
        initial_equity=initial_equity,
        n_windows=max(1, min(n_windows, 12)),
    )
    result = await loop.run_in_executor(None, engine.run)
    return result.to_dict()


class _ResetRequest(BaseModel):
    pin: str
    tracks: list[str] | None = None


@app.post("/api/reset")
async def reset_simulation(body: _ResetRequest):
    """
    Reset simulation state for the given tracks (default: all).
    Requires the correct PIN. Clears in-memory portfolios and all heuristic files.
    """
    import src.agent.memory as _memory
    import shutil

    if not secrets.compare_digest(str(body.pin), settings.reset_pin):
        return {"error": "Invalid PIN"}

    target_tracks = body.tracks if body.tracks else list(settings.tracks)
    invalid = [t for t in target_tracks if t not in settings.tracks]
    if invalid:
        return {"error": f"Unknown tracks: {invalid}"}

    # Never reset under a running scan — the scan holds references to the old
    # portfolios and its end-of-scan persist would resurrect the cleared state.
    from src.scheduler.scan_loop import _scan_lock
    if not _scan_lock.acquire(blocking=False):
        return {"error": "Scan in progress — try again in a moment"}
    try:
        from src.db import Decision, get_session

        cleared: dict = {}
        for track in target_tracks:
            # Count and delete heuristic files
            heuristic_dir = settings.heuristics_dir / track
            heuristic_count = len(list(heuristic_dir.glob("*.json"))) if heuristic_dir.exists() else 0
            if heuristic_dir.exists():
                shutil.rmtree(heuristic_dir)
                heuristic_dir.mkdir(parents=True, exist_ok=True)

            # Clear cached heuristic store so next call rebuilds from empty dir
            _memory._stores.pop(track, None)

            # Delete persisted decisions for this track
            db = get_session()
            try:
                decision_count = db.query(Decision).filter(Decision.track == track).delete()
                db.commit()
            finally:
                db.close()

            cleared[track] = {"heuristics_deleted": heuristic_count, "decisions_deleted": decision_count}

        # Reset in-memory portfolios, drop their persisted state (so a restart
        # doesn't resurrect them), and clear the latest-decisions cache.
        from src.portfolio.persistence import delete_portfolio_state
        reset_portfolios(target_tracks)
        delete_portfolio_state(target_tracks)
        clear_recent_decisions()
    finally:
        _scan_lock.release()

    await _broadcast({"event": "simulation_reset", "data": {"tracks": target_tracks}})
    logger.info("Simulation reset for tracks: %s", target_tracks)
    return {"reset": True, "tracks": target_tracks, "cleared": cleared}


class _BackfillRequest(BaseModel):
    pin: str
    tracks: list[str] | None = None


@app.post("/api/erl/backfill")
async def erl_backfill(body: _BackfillRequest):
    """Re-run ERL over closed trades that have no heuristic yet (e.g. trades that
    closed while the Claude ERL call was failing). PIN-guarded — it spends model
    tokens — and offloaded to a worker thread since ERL is a long blocking call.
    Idempotent: trades that already sourced a heuristic are skipped."""
    if not secrets.compare_digest(str(body.pin), settings.reset_pin):
        return {"error": "Invalid PIN"}
    target = body.tracks if body.tracks else list(settings.tracks)
    invalid = [t for t in target if t not in settings.tracks]
    if invalid:
        return {"error": f"Unknown tracks: {invalid}"}

    from src.scheduler.scan_loop import backfill_erl
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, backfill_erl, target)
    logger.info("ERL backfill complete: %s", result)
    return {"backfilled": True, "result": result}


@app.post("/api/scan/{market}")
async def trigger_scan(market: str):
    """Manually trigger a scan. run_scan is a long blocking call, so it runs in a
    worker thread — otherwise it would freeze the whole event loop (every page and
    API refresh) for the duration of the scan."""
    if market not in SCAN_MARKETS:
        return {"error": f"market must be one of {SCAN_MARKETS}"}
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_scan, market)
    await _broadcast({"event": "scan_complete", "data": result})
    return result


@app.get("/api/debug/screener/{market}")
async def debug_screener(market: str):
    """Return per-ticker screener breakdown — why each stock passed or was rejected."""
    if market not in SCAN_MARKETS:
        return {"error": f"market must be one of {SCAN_MARKETS}"}

    from src.data.market_data import fetch_batch_eu, fetch_batch_nordic, fetch_batch_us, get_sector
    from src.data.watchlist import get_eu_watchlist, get_omxs30_tickers, get_us_tickers
    from src.analysis.technical import compute_signals
    from src.analysis.regime import classify_regime
    from src.scheduler.scan_loop import _to_sek_price
    from config.settings import settings as s

    if market == "nordic":
        fetch_fn, watchlist = fetch_batch_nordic, get_omxs30_tickers()
    elif market == "eu":
        fetch_fn, watchlist = fetch_batch_eu, get_eu_watchlist()
    else:
        fetch_fn, watchlist = fetch_batch_us, get_us_tickers()
    batch = fetch_fn(watchlist)

    rows = []
    for ticker, df in batch.items():
        if df is None or df.empty:
            rows.append({"ticker": ticker, "status": "no_data"})
            continue
        try:
            signals = compute_signals(ticker, df)
            regime  = classify_regime(df)
            reasons = []
            if not signals.price_above_50sma:
                reasons.append("below_50sma")
            if not (s.rsi_min <= signals.rsi_14 <= s.rsi_max):
                reasons.append(f"rsi_{signals.rsi_14:.1f}_outside_{s.rsi_min}-{s.rsi_max}")
            if signals.volume_ratio < s.volume_spike_multiplier:
                reasons.append(f"vol_{signals.volume_ratio:.2f}x<{s.volume_spike_multiplier}x")
            if regime.regime == "neutral":
                reasons.append("neutral_regime")
            elif regime.regime == "trending" and not signals.ema_21_above_50sma:
                reasons.append("ema21_below_sma50")
            elif regime.regime == "mean-reverting" and signals.bb_pct_b > 0.35:
                reasons.append(f"bb_pct_b_{signals.bb_pct_b:.2f}>0.35")
            rows.append({
                "ticker": ticker,
                "status": "PASS" if not reasons else "REJECT",
                "reasons": reasons,
                "rsi": round(signals.rsi_14, 1),
                "volume_ratio": round(signals.volume_ratio, 2),
                "regime": regime.regime,
                "price_above_50sma": signals.price_above_50sma,
                "ema21_above_50sma": signals.ema_21_above_50sma,
            })
        except Exception as e:
            rows.append({"ticker": ticker, "status": "error", "error": str(e)})

    passed  = [r for r in rows if r.get("status") == "PASS"]
    reasons_tally: dict[str, int] = {}
    for r in rows:
        for reason in r.get("reasons", []):
            key = reason.split("_")[0] if "_" in reason else reason
            reasons_tally[key] = reasons_tally.get(key, 0) + 1

    return {"market": market, "total": len(rows), "passed": len(passed), "rejection_tally": reasons_tally, "tickers": rows}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # BaseHTTPMiddleware only sees http-scope requests — websocket connections
    # bypass _AuthMiddleware entirely, so auth must be enforced here.
    if settings.dashboard_password and not _token_valid(websocket.cookies.get(_SESSION_COOKIE, "")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


async def _broadcast(data: Any) -> None:
    """Push JSON data to all connected WebSocket clients."""
    message = json.dumps(data)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


@app.get("/api/prompts")
async def prompts():
    """Current and historical DSPy instructions for each track, as saved by MIPRO."""
    try:
        from src.agent.decision import TradeDecision
        baseline = (TradeDecision.__doc__ or "").strip()
    except Exception:
        baseline = "Baseline instructions unavailable."

    result = {}
    for track in settings.tracks:
        compiled_dir = settings.compiled_dir
        current_path = compiled_dir / f"{track}_trade_decision.json"

        current = None
        if current_path.exists():
            state = _parse_dspy_json(current_path)
            mtime = datetime.utcfromtimestamp(os.path.getmtime(current_path))
            current = {
                "instructions": state.get("instructions", ""),
                "demos_count": len(state.get("demos", [])),
                "timestamp": mtime.strftime("%Y-%m-%d %H:%M UTC"),
            }

        history = []
        if compiled_dir.exists():
            for p in sorted(compiled_dir.glob(f"{track}_trade_decision_*.json"), reverse=True):
                state = _parse_dspy_json(p)
                history.append({
                    "instructions": state.get("instructions", ""),
                    "demos_count": len(state.get("demos", [])),
                    "timestamp": _ts_from_archive_name(p.name),
                    "filename": p.name,
                })

        result[track] = {"baseline": baseline, "current": current, "history": history}

    return result


def _parse_dspy_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    instructions = raw.get("signature_instructions") or _find_json_key(raw, "instructions") or ""
    demos = raw.get("demos") or _find_json_key(raw, "demos") or []
    return {"instructions": str(instructions), "demos": demos if isinstance(demos, list) else []}


def _find_json_key(data, key):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            found = _find_json_key(v, key)
            if found is not None:
                return found
    return None


def _ts_from_archive_name(name: str) -> str:
    m = re.search(r"_(\d{8})_(\d{6})\.json$", name)
    if not m:
        return "Unknown"
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]} UTC"

