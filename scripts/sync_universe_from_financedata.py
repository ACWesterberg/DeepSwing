#!/usr/bin/env python3
"""Build DeepSwing universe CSVs from the FinanceData broker universe cache."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

NORDIC_COUNTRIES = frozenset({"Sweden", "Norway", "Finland", "Denmark"})
NORDIC_EXCHANGE_CODES = frozenset({"ST", "OL", "HE", "CO"})
US_LIT_EXCHANGES = frozenset({"NYSE", "NASDAQ"})

EXCHANGE_CODE_TO_LABEL: dict[str, str] = {
    "ST": "OMXS",
    "OL": "OSLO",
    "HE": "OMXH",
    "CO": "OMXC",
    "NYSE": "NYSE",
    "NASDAQ": "NASDAQ",
}

COUNTRY_TO_CODE: dict[str, str] = {
    "Sweden": "SE",
    "Norway": "NO",
    "Finland": "FI",
    "Denmark": "DK",
    "United States": "US",
}

YAHOO_SUFFIX: dict[str, str] = {
    "ST": "ST",
    "OL": "OL",
    "HE": "HE",
    "CO": "CO",
}

FIELDNAMES = ("name", "yahoo_ticker", "isin", "country", "exchange", "sector", "enabled")


def _load_financedata_rows(source: str | None) -> list[dict]:
    if source:
        path = Path(source)
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    try:
        from financedata import get_universe

        return get_universe(active_only=True)
    except ImportError:
        snapshot = _BASE.parent / "FinanceData" / "data" / "universe_snapshot.csv"
        if snapshot.is_file():
            with open(snapshot, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        raise SystemExit(
            "financedata is not installed and no --source CSV was given. "
            "Install with `pip install -e ../FinanceData` or pass --source."
        )


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["yahoo_ticker"]: row for row in csv.DictReader(f)}


def _yahoo_ticker(row: dict) -> str | None:
    code = (row.get("exchange_code") or "").strip()
    ticker = (row.get("ticker") or "").strip()
    if not ticker:
        return None
    if code in YAHOO_SUFFIX:
        return f"{ticker}.{YAHOO_SUFFIX[code]}"
    if code == "US":
        return ticker
    return None


def _deepswing_exchange(row: dict) -> str | None:
    code = (row.get("exchange_code") or "").strip()
    if code in EXCHANGE_CODE_TO_LABEL:
        return EXCHANGE_CODE_TO_LABEL[code]
    if code == "US":
        venue = (row.get("exchange") or "").strip()
        return venue if venue in US_LIT_EXCHANGES else None
    return None


def _convert_row(row: dict, existing: dict[str, dict]) -> dict | None:
    if (row.get("status") or "active").strip().lower() != "active":
        return None

    country = (row.get("country") or "").strip()
    code = (row.get("exchange_code") or "").strip()
    yahoo = _yahoo_ticker(row)
    exchange = _deepswing_exchange(row)
    if not yahoo or not exchange:
        return None

    prev = existing.get(yahoo, {})
    enabled = prev.get("enabled", "true")
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() == "true"
    else:
        enabled = bool(enabled)

    return {
        "name": (row.get("company_name") or prev.get("name") or "").strip(),
        "yahoo_ticker": yahoo,
        "isin": (row.get("isin") or prev.get("isin") or "").strip(),
        "country": COUNTRY_TO_CODE.get(country, prev.get("country", "")),
        "exchange": exchange,
        "sector": (prev.get("sector") or "").strip(),
        "enabled": "true" if enabled else "false",
    }


def _nordic_row(row: dict) -> bool:
    country = (row.get("country") or "").strip()
    code = (row.get("exchange_code") or "").strip()
    return country in NORDIC_COUNTRIES and code in NORDIC_EXCHANGE_CODES


def _us_row(row: dict) -> bool:
    if (row.get("country") or "").strip() != "United States":
        return False
    if (row.get("exchange_code") or "").strip() != "US":
        return False
    return (row.get("exchange") or "").strip() in US_LIT_EXCHANGES


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["country"], r["exchange"], r["yahoo_ticker"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def sync_universe(
    *,
    source: str | None = None,
    nordic_path: Path | None = None,
    us_path: Path | None = None,
) -> tuple[int, int]:
    nordic_path = nordic_path or _BASE / "config" / "universe.csv"
    us_path = us_path or _BASE / "config" / "universe_global.csv"

    raw_rows = _load_financedata_rows(source)
    existing_nordic = _load_existing(nordic_path)
    existing_us = _load_existing(us_path)
    existing = {**existing_us, **existing_nordic}

    nordic_out: list[dict] = []
    us_out: list[dict] = []

    for raw in raw_rows:
        if _nordic_row(raw):
            converted = _convert_row(raw, existing)
            if converted:
                nordic_out.append(converted)
        if _us_row(raw):
            converted = _convert_row(raw, existing)
            if converted:
                us_out.append(converted)

    _write_csv(nordic_path, nordic_out)
    _write_csv(us_path, us_out)
    return len(nordic_out), len(us_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="FinanceData universe CSV export (default: financedata cache or ../FinanceData snapshot)",
    )
    parser.add_argument("--nordic-out", type=Path, default=_BASE / "config" / "universe.csv")
    parser.add_argument("--us-out", type=Path, default=_BASE / "config" / "universe_global.csv")
    args = parser.parse_args(argv)

    nordic_n, us_n = sync_universe(
        source=args.source,
        nordic_path=args.nordic_out,
        us_path=args.us_out,
    )
    print(f"Wrote {nordic_n:,} Nordic rows to {args.nordic_out}")
    print(f"Wrote {us_n:,} US rows to {args.us_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
