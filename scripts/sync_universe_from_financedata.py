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

EU_COUNTRIES = frozenset({
    "United Kingdom", "Germany", "France", "Netherlands", "Belgium", "Spain",
    "Switzerland", "Poland", "Austria", "Portugal", "Ireland",
})
EU_EXCHANGE_CODES = frozenset({"LSE", "XETRA", "PA", "AS", "BR", "MC", "SW", "WAR", "VI", "LS", "IR"})

US_LIT_EXCHANGES = frozenset({"NYSE", "NASDAQ"})

NORDIC_EXCHANGE_CODE_TO_LABEL: dict[str, str] = {
    "ST": "OMXS",
    "OL": "OSLO",
    "HE": "OMXH",
    "CO": "OMXC",
}

EU_EXCHANGE_CODE_TO_LABEL: dict[str, str] = {
    "LSE": "LSE",
    "XETRA": "XETRA",
    "PA": "EURONEXT",
    "AS": "EURONEXT",
    "BR": "EURONEXT",
    "MC": "BME",
    "SW": "SIX",
    "WAR": "WSE",
    "VI": "VIE",
    "LS": "EURONEXT",
    "IR": "EURONEXT",
}

NORDIC_YAHOO_SUFFIX: dict[str, str] = {
    "ST": "ST",
    "OL": "OL",
    "HE": "HE",
    "CO": "CO",
}

EU_YAHOO_SUFFIX: dict[str, str] = {
    "LSE": "L",
    "XETRA": "DE",
    "PA": "PA",
    "AS": "AS",
    "BR": "BR",
    "MC": "MC",
    "SW": "SW",
    "WAR": "WA",
    "VI": "VI",
    "LS": "LS",
    "IR": "IR",
}

COUNTRY_TO_CODE: dict[str, str] = {
    "Sweden": "SE",
    "Norway": "NO",
    "Finland": "FI",
    "Denmark": "DK",
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "France": "FR",
    "Netherlands": "NL",
    "Belgium": "BE",
    "Spain": "ES",
    "Switzerland": "CH",
    "Poland": "PL",
    "Austria": "AT",
    "Portugal": "PT",
    "Ireland": "IE",
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


def _yahoo_ticker(row: dict, suffix_map: dict[str, str]) -> str | None:
    code = (row.get("exchange_code") or "").strip()
    ticker = (row.get("ticker") or "").strip()
    if not ticker:
        return None
    suffix = suffix_map.get(code)
    if suffix:
        return f"{ticker}.{suffix}"
    if code == "US":
        return ticker
    return None


def _deepswing_exchange(row: dict, label_map: dict[str, str]) -> str | None:
    code = (row.get("exchange_code") or "").strip()
    if code in label_map:
        return label_map[code]
    if code == "US":
        venue = (row.get("exchange") or "").strip()
        return venue if venue in US_LIT_EXCHANGES else None
    return None


def _convert_row(
    row: dict,
    existing: dict[str, dict],
    *,
    suffix_map: dict[str, str],
    label_map: dict[str, str],
) -> dict | None:
    if (row.get("status") or "active").strip().lower() != "active":
        return None

    country = (row.get("country") or "").strip()
    yahoo = _yahoo_ticker(row, suffix_map)
    exchange = _deepswing_exchange(row, label_map)
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


def _eu_row(row: dict) -> bool:
    country = (row.get("country") or "").strip()
    code = (row.get("exchange_code") or "").strip()
    return country in EU_COUNTRIES and code in EU_EXCHANGE_CODES


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
    eu_path: Path | None = None,
    us_path: Path | None = None,
) -> tuple[int, int, int]:
    nordic_path = nordic_path or _BASE / "config" / "universe.csv"
    eu_path = eu_path or _BASE / "config" / "universe_eu.csv"
    us_path = us_path or _BASE / "config" / "universe_global.csv"

    raw_rows = _load_financedata_rows(source)
    existing_nordic = _load_existing(nordic_path)
    existing_eu = _load_existing(eu_path)
    existing_us = _load_existing(us_path)
    existing = {**existing_us, **existing_eu, **existing_nordic}

    nordic_out: list[dict] = []
    eu_out: list[dict] = []
    us_out: list[dict] = []

    for raw in raw_rows:
        if _nordic_row(raw):
            converted = _convert_row(
                raw, existing,
                suffix_map=NORDIC_YAHOO_SUFFIX,
                label_map=NORDIC_EXCHANGE_CODE_TO_LABEL,
            )
            if converted:
                nordic_out.append(converted)
        if _eu_row(raw):
            converted = _convert_row(
                raw, existing,
                suffix_map=EU_YAHOO_SUFFIX,
                label_map=EU_EXCHANGE_CODE_TO_LABEL,
            )
            if converted:
                eu_out.append(converted)
        if _us_row(raw):
            converted = _convert_row(
                raw, existing,
                suffix_map={},
                label_map={},
            )
            if converted:
                us_out.append(converted)

    _write_csv(nordic_path, nordic_out)
    _write_csv(eu_path, eu_out)
    _write_csv(us_path, us_out)
    return len(nordic_out), len(eu_out), len(us_out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        help="FinanceData universe CSV export (default: financedata cache or ../FinanceData snapshot)",
    )
    parser.add_argument("--nordic-out", type=Path, default=_BASE / "config" / "universe.csv")
    parser.add_argument("--eu-out", type=Path, default=_BASE / "config" / "universe_eu.csv")
    parser.add_argument("--us-out", type=Path, default=_BASE / "config" / "universe_global.csv")
    args = parser.parse_args(argv)

    nordic_n, eu_n, us_n = sync_universe(
        source=args.source,
        nordic_path=args.nordic_out,
        eu_path=args.eu_out,
        us_path=args.us_out,
    )
    print(f"Wrote {nordic_n:,} Nordic rows to {args.nordic_out}")
    print(f"Wrote {eu_n:,} EU rows to {args.eu_out}")
    print(f"Wrote {us_n:,} US rows to {args.us_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
