from __future__ import annotations

import hashlib
import html
import logging
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config.settings import settings
from src.agent.watch_classifier import classify_watch_event
from src.data.insider_fetcher import get_insider_summary
from src.data.market_data import fetch_ohlcv, get_current_price
from src.data.news_fetcher import fetch_news_for_ticker
from src.db import WatchAlert, WatchedTicker, get_session
from src.notify.telegram import send_telegram
from src.scheduler.market_hours import is_market_open

logger = logging.getLogger(__name__)

_SEEN_HASHES_KEPT = 400


def run_watch_monitor() -> dict:
    """Check every watched ticker for large day moves, fresh directional news and
    insider activity; ping Telegram for non-neutral events. Dedupe state lives on
    the WatchedTicker row, committed per ticker so a mid-run crash never re-pings."""
    session = get_session()
    alerts: list[dict] = []
    try:
        watches = session.query(WatchedTicker).all()
        if not watches:
            return {"checked": 0, "alerts": []}

        for watch in watches:
            try:
                alerts.extend(_check_ticker(session, watch))
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.error("Watch monitor error for %s: %s", watch.ticker, exc, exc_info=True)

        _prune_alerts(session)
        if alerts:
            logger.info("Watch monitor: %d alert(s) across %d ticker(s)", len(alerts), len(watches))
        return {"checked": len(watches), "alerts": alerts}
    finally:
        session.close()


def _check_ticker(session, watch: WatchedTicker) -> list[dict]:
    alerts: list[dict] = []
    baseline = not watch.baselined
    ticker, market = watch.ticker, watch.market

    live = get_current_price(ticker, market)
    move: Optional[float] = None
    if live is not None:
        prev_close = _previous_close(ticker, market)
        if prev_close:
            move = live / prev_close - 1.0
        watch.last_price = float(live)
        watch.last_move_pct = move
    watch.last_checked = datetime.utcnow()

    # Large day move — only while the exchange trades (off-hours quotes are stale)
    if move is not None and not baseline and is_market_open(market):
        today = _local_today(market).isoformat()
        if _move_alert_due(move, today, watch.last_alert_day, watch.last_alert_move_pct):
            verdict = "bullish" if move > 0 else "bearish"
            arrow = "📈" if move > 0 else "📉"
            msg = f"{arrow} <b>{html.escape(ticker)}</b> {move:+.1%} today ({live:.2f})"
            alerts.append(_record_alert(session, ticker, "move", verdict, msg))
            watch.last_alert_day = today
            watch.last_alert_move_pct = move

    # Fresh news — classified in one shared cheap call; neutral stays silent
    articles = fetch_news_for_ticker(ticker, market)
    fresh = [a for a in articles if _recent_enough(a)]
    seen = set(watch.seen_news_hashes or [])
    new_items = [(h, a) for a in fresh if (h := _news_hash(a)) not in seen]
    if new_items:
        if not baseline:
            headlines = "\n".join(f"- {a.get('headline', '')}" for _, a in new_items[:10])
            verdict, reason = classify_watch_event(ticker, "news headlines", headlines)
            if verdict != "neutral":
                lines = "\n".join(
                    f"• {html.escape(a.get('headline', ''))}" for _, a in new_items[:5]
                )
                msg = (
                    f"📰 <b>{html.escape(ticker)}</b> — {verdict.upper()} news\n"
                    f"{html.escape(reason)}\n{lines}"
                )
                alerts.append(_record_alert(session, ticker, "news", verdict, msg))
        ordered = (watch.seen_news_hashes or []) + [h for h, _ in new_items]
        watch.seen_news_hashes = ordered[-_SEEN_HASHES_KEPT:]

    # Insider activity — alert when the summary text changes
    insider = get_insider_summary(ticker, market) or ""
    insider_hash = _sha(insider)
    if insider_hash != watch.insider_hash:
        if not baseline and insider.strip():
            verdict, reason = classify_watch_event(ticker, "insider activity", insider)
            skip_sell = settings.watch_insider_buys_only and verdict == "bearish"
            if verdict != "neutral" and not skip_sell:
                msg = (
                    f"👤 <b>{html.escape(ticker)}</b> — {verdict.upper()} insider activity\n"
                    f"{html.escape(reason)}\n{html.escape(insider.strip()[:400])}"
                )
                alerts.append(_record_alert(session, ticker, "insider", verdict, msg))
        watch.insider_hash = insider_hash

    if baseline:
        watch.baselined = True
    return alerts


def _record_alert(session, ticker: str, kind: str, verdict: str, message: str) -> dict:
    delivered = send_telegram(message)
    row = WatchAlert(ticker=ticker, kind=kind, verdict=verdict, message=message, delivered=delivered)
    session.add(row)
    logger.info("Watch alert [%s/%s] %s (delivered=%s)", kind, verdict, ticker, delivered)
    return {"ticker": ticker, "kind": kind, "verdict": verdict, "message": message, "delivered": delivered}


def _move_alert_due(move: float, today: str, last_day: Optional[str], last_move: Optional[float]) -> bool:
    """First ping of the day at the threshold; re-ping only when the move extends
    a full step beyond the last alerted level, or flips direction."""
    if abs(move) < settings.watch_move_alert_pct:
        return False
    if last_day != today or last_move is None:
        return True
    if (move > 0) != (last_move > 0):
        return True
    return abs(move) - abs(last_move) >= settings.watch_move_realert_step_pct


def _previous_close(ticker: str, market: str) -> Optional[float]:
    """Last *completed* daily close — intraday feeds append today's forming bar,
    which would make the day-move read ~0%."""
    df = fetch_ohlcv(ticker, market, period="1mo")
    if df is None or len(df) == 0:
        return None
    closes = df["Close"]
    try:
        last_bar_date = pd.Timestamp(df.index[-1]).date()
    except (TypeError, ValueError):
        last_bar_date = None
    if last_bar_date == _local_today(market) and len(closes) >= 2:
        return float(closes.iloc[-2])
    return float(closes.iloc[-1])


def _local_today(market: str) -> date:
    tz = ZoneInfo("America/New_York") if market == "us" else ZoneInfo("Europe/Stockholm")
    return datetime.now(tz).date()


def _recent_enough(article: dict) -> bool:
    published = article.get("published_at")
    if not published:
        return True
    try:
        ts = pd.Timestamp(published)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_hours = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600
        return age_hours <= settings.watch_news_max_age_hours
    except (TypeError, ValueError):
        return True


def _news_hash(article: dict) -> str:
    return _sha(f"{article.get('headline', '')}|{article.get('source_url', '')}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _prune_alerts(session) -> None:
    keep = settings.watch_alerts_retention
    if keep <= 0:
        return
    try:
        cutoff_row = (
            session.query(WatchAlert.id)
            .order_by(WatchAlert.id.desc())
            .offset(keep)
            .first()
        )
        if cutoff_row:
            session.query(WatchAlert).filter(WatchAlert.id <= cutoff_row[0]).delete(
                synchronize_session=False
            )
            session.commit()
    except Exception as exc:
        session.rollback()
        logger.warning("Watch alert prune failed: %s", exc)
