from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_insider_cache: dict[str, tuple[datetime, str]] = {}
_CACHE_TTL_HOURS = 24


def get_insider_summary(ticker: str, market: str) -> str:
    """Return a short insider activity summary for the ticker."""
    cache_key = f"{market}:{ticker}"
    cached = _insider_cache.get(cache_key)
    if cached:
        ts, text = cached
        if (datetime.utcnow() - ts).total_seconds() < _CACHE_TTL_HOURS * 3600:
            return text

    if market == "us":
        text = _fetch_sec_edgar(ticker)
    else:
        text = _fetch_fi_insynsregistret(ticker)

    _insider_cache[cache_key] = (datetime.utcnow(), text)
    return text


def _fetch_sec_edgar(ticker: str) -> str:
    """Fetch recent Form 4 filings from SEC EDGAR (free public API)."""
    try:
        # CIK lookup
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={_days_ago(30)}&forms=4"
        resp = httpx.get(search_url, timeout=10, headers={"User-Agent": "DeepSwing/1.0 contact@deepswing.local"})
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return f"No recent insider filings found for {ticker} (SEC EDGAR)."

        summaries = []
        for hit in hits[:5]:
            source = hit.get("_source", {})
            summaries.append(
                f"{source.get('file_date', '?')}: "
                f"{source.get('display_names', ['?'])[0]} — "
                f"Form 4 filing"
            )

        return "SEC Insider Activity:\n" + "\n".join(summaries)

    except Exception as exc:
        logger.debug("SEC EDGAR error for %s: %s", ticker, exc)
        return f"Insider data unavailable for {ticker}."


def _fetch_fi_insynsregistret(ticker: str) -> str:
    """
    Fetch from Finansinspektionen's insynsregister (Swedish insider registry).
    FI publishes a CSV export at a public URL.
    """
    try:
        url = "https://fi.se/contentassets/2c7b86aa49b74e37b3eb8fe91da5ccbc/insynsregistret.csv"
        resp = httpx.get(url, timeout=15, headers={"User-Agent": "DeepSwing/1.0"})
        resp.raise_for_status()

        ticker_base = ticker.split(".")[0].upper()
        reader = csv.DictReader(io.StringIO(resp.text), delimiter=";")

        since = datetime.utcnow() - timedelta(days=30)
        matches = []
        for row in reader:
            issuer = (row.get("Emittent", "") or row.get("Issuer", "")).upper()
            if ticker_base not in issuer:
                continue
            try:
                trade_date_str = row.get("Handelsdatum", "") or row.get("TransactionDate", "")
                trade_date = datetime.strptime(trade_date_str[:10], "%Y-%m-%d")
                if trade_date >= since:
                    matches.append(row)
            except (ValueError, KeyError):
                pass

        if not matches:
            return f"No recent insider activity for {ticker_base} (FI register)."

        summaries = []
        for row in matches[:5]:
            person = row.get("Person", row.get("Insider", "?"))
            trans = row.get("Transaktionstyp", row.get("TransactionType", "?"))
            volume = row.get("Volym", row.get("Volume", "?"))
            price = row.get("Kurs", row.get("Price", "?"))
            summaries.append(f"{person}: {trans} — {volume} shares @ {price}")

        return "FI Insider Activity:\n" + "\n".join(summaries)

    except Exception as exc:
        logger.debug("FI Insynsregistret error for %s: %s", ticker, exc)
        return f"Nordic insider data unavailable for {ticker}."


def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")
