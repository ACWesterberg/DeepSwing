from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from config.settings import settings
from src.portfolio.metrics import compute_metrics
from src.portfolio.simulator import get_portfolio
from src.scheduler.market_hours import active_markets, is_market_open
from src.scheduler.scan_loop import run_scan, set_trade_event_handler

logger = logging.getLogger(__name__)

app = FastAPI(title="DeepSwing Dashboard", version="1.0.0")

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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/status")
async def status():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "active_markets": active_markets(),
        "nordic_open": is_market_open("nordic"),
        "us_open": is_market_open("us"),
        "tracks": settings.tracks,
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
            "equity_curve": _build_equity_curve_data(portfolio),
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


@app.post("/api/scan/{market}")
async def trigger_scan(market: str):
    """Manually trigger a scan (for testing)."""
    if market not in ("nordic", "us"):
        return {"error": "market must be 'nordic' or 'us'"}
    result = run_scan(market)
    await _broadcast({"event": "scan_complete", "data": result})
    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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


def _build_equity_curve_data(portfolio) -> list[dict]:
    """Build equity curve as [{date, equity}] list for Chart.js."""
    equity = portfolio.starting_equity
    points = [{"date": "start", "equity": equity}]
    for trade in sorted(portfolio.closed_trades, key=lambda t: t.exit_time):
        equity += trade.pnl
        points.append({
            "date": trade.exit_time.strftime("%Y-%m-%d %H:%M"),
            "equity": round(equity, 2),
        })
    return points
