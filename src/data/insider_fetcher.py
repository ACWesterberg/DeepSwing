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


_FI_COLUMN_ALIASES: dict[str, list[str]] = {
    "issuer":   ["Emittent", "Issuer", "emittent", "issuer", "Bolag"],
    "date":     ["Handelsdatum", "TransactionDate", "Datum", "Date", "Transaction date"],
    "person":   ["Person", "Insider", "Namn", "Name"],
    "type":     ["Transaktionstyp", "TransactionType", "Typ", "Type", "Transaction type"],
    "volume":   ["Volym", "Volume", "Antal", "Quantity"],
    "price":    ["Kurs", "Price", "Pris"],
}


def _resolve_col(row: dict, canonical: str) -> str:
    """Return the first non-empty value found across all aliases for a canonical column."""
    for alias in _FI_COLUMN_ALIASES.get(canonical, []):
        val = row.get(alias, "")
        if val:
            return val.strip()
    return "?"


def _detect_delimiter(text: str) -> str:
    """Guess CSV delimiter from the first non-empty line."""
    for line in text.splitlines():
        if line.strip():
            return ";" if line.count(";") >= line.count(",") else ","
    return ";"


def _decode_fi_response(content: bytes) -> str:
    """Try UTF-8 then Latin-1; strip BOM."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def _fetch_fi_insynsregistret(ticker: str) -> str:
    """
    Fetch from Finansinspektionen's insynsregister (Swedish insider registry).
    FI publishes a CSV export at a public URL — column names vary by export version.
    """
    try:
        url = "https://fi.se/contentassets/2c7b86aa49b74e37b3eb8fe91da5ccbc/insynsregistret.csv"
        resp = httpx.get(url, timeout=15, headers={"User-Agent": "DeepSwing/1.0"})
        resp.raise_for_status()

        text = _decode_fi_response(resp.content)
        delimiter = _detect_delimiter(text)
        ticker_base = ticker.split(".")[0].upper()
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)

        since = datetime.utcnow() - timedelta(days=30)
        matches = []
        for row in reader:
            issuer = _resolve_col(row, "issuer").upper()
            if ticker_base not in issuer:
                continue
            date_str = _resolve_col(row, "date")
            try:
                trade_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
                if trade_date >= since:
                    matches.append(row)
            except ValueError:
                pass

        if not matches:
            return f"No recent insider activity for {ticker_base} (FI register)."

        summaries = []
        for row in matches[:5]:
            person = _resolve_col(row, "person")
            trans = _resolve_col(row, "type")
            volume = _resolve_col(row, "volume")
            price = _resolve_col(row, "price")
            summaries.append(f"{person}: {trans} — {volume} shares @ {price}")

        return "FI Insider Activity:\n" + "\n".join(summaries)

    except Exception as exc:
        logger.debug("FI Insynsregistret error for %s: %s", ticker, exc)
        return f"Nordic insider data unavailable for {ticker}."


def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%d")
