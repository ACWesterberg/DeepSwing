from __future__ import annotations

import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


def telegram_configured() -> bool:
    return bool(settings.telegram_bot_token and settings.telegram_chat_id)


def send_telegram(text: str) -> bool:
    """Send one message to the configured chat. Returns delivery success; a
    missing token/chat_id logs and returns False so alerting stays dormant
    until the user drops the keys into .env."""
    if not telegram_configured():
        logger.info("Telegram not configured — alert not sent: %s", text[:120])
        return False
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
            json={
                "chat_id": settings.telegram_chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10.0,
        )
        if resp.status_code != 200:
            logger.warning("Telegram send failed (%d): %s", resp.status_code, resp.text[:200])
            return False
        return True
    except Exception as exc:
        logger.warning("Telegram send error: %s", exc)
        return False
