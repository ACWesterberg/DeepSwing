#!/usr/bin/env python3
"""
Verify the two event data sources against the live APIs.

The Kalshi and NWS clients were written from documented schemas without live
access, so their field names and base URL are unconfirmed. Run this once on the
Pi before trusting anything the events track produces:

    venv/bin/python scripts/check_event_sources.py

It prints what each API actually returned, flags any field the parser expected
but did not find, and tries the alternative Kalshi host if the configured one
fails. Read-only — it opens no positions and needs no credentials.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config.settings import settings  # noqa: E402
from src.analysis.event_model import bucket_bounds, build_candidates  # noqa: E402
from src.data import kalshi, weather_forecast  # noqa: E402

KALSHI_HOSTS = [
    settings.kalshi_api_base,
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://external-api.kalshi.com/trade-api/v2",
]

EXPECTED_MARKET_FIELDS = [
    "ticker", "event_ticker", "title", "yes_bid", "yes_ask", "last_price",
    "volume", "open_interest", "close_time", "strike_type",
    "floor_strike", "cap_strike",
]


def ok(msg: str) -> None:
    print(f"  \033[32mOK\033[0m   {msg}")


def bad(msg: str) -> None:
    print(f"  \033[31mFAIL\033[0m {msg}")


def warn(msg: str) -> None:
    print(f"  \033[33mWARN\033[0m {msg}")


def check_kalshi() -> tuple[str | None, list[dict]]:
    """Find a working Kalshi host and return it with a page of raw markets."""
    print("\n=== Kalshi market data ===")
    series = settings.event_series[0] if settings.event_series else "KXHIGHNY"

    for host in dict.fromkeys(KALSHI_HOSTS):
        url = f"{host.rstrip('/')}/markets"
        try:
            with httpx.Client(timeout=settings.kalshi_timeout_seconds) as client:
                response = client.get(
                    url, params={"series_ticker": series, "status": "open", "limit": 5}
                )
                response.raise_for_status()
                markets = response.json().get("markets") or []
        except Exception as exc:
            bad(f"{host} -> {type(exc).__name__}: {exc}")
            continue

        ok(f"{host} returned {len(markets)} open {series} markets")
        if host != settings.kalshi_api_base:
            warn(f"Set KALSHI_API_BASE={host} in .env — it is not the configured host")
        return host, markets

    bad("No Kalshi host reachable. Check egress and the series ticker.")
    return None, []


def check_market_schema(markets: list[dict]) -> None:
    print("\n=== Market schema ===")
    if not markets:
        warn("No markets to inspect — is the series ticker right, and is it in season?")
        return

    sample = markets[0]
    print(f"  Raw first market:\n{json.dumps(sample, indent=4)[:1200]}")

    missing = [f for f in EXPECTED_MARKET_FIELDS if f not in sample]
    if missing:
        bad(f"Fields the parser expects but the API did not return: {missing}")
        warn("Update _parse_market in src/data/kalshi.py to the real names above.")
    else:
        ok("Every expected field is present")

    strike_types = {(m.get("strike_type") or "?").lower() for m in markets}
    print(f"  strike_type values seen: {sorted(strike_types)}")
    unknown = strike_types - {
        "between", "greater", "greater_or_equal", "less", "less_or_equal", "?"
    }
    if unknown:
        bad(f"Unhandled strike types: {sorted(unknown)} — bucket_bounds will reject these")

    parsed = [kalshi._parse_market(m, m.get("ticker", "").split("-")[0]) for m in markets]
    usable = [c for c in parsed if c is not None]
    ok(f"Parsed {len(usable)}/{len(markets)} markets into contracts")

    for contract in usable[:3]:
        try:
            lo, hi = bucket_bounds(contract)
            print(
                f"    {contract.ticker}: bid {contract.yes_bid:.2f} ask {contract.yes_ask:.2f} "
                f"oi {contract.open_interest} -> YES on [{lo}, {hi}] degF"
            )
        except ValueError as exc:
            bad(f"    {contract.ticker}: {exc}")


def check_nws() -> None:
    print("\n=== NWS forecasts ===")
    for series in settings.event_series:
        station = weather_forecast.get_station(series)
        if station is None:
            bad(f"{series}: no station configured")
            continue
        try:
            highs = weather_forecast._forecast_highs_for_series(series)
        except Exception as exc:
            bad(f"{series} ({station.name}): {type(exc).__name__}: {exc}")
            continue
        if not highs:
            bad(f"{series} ({station.name}): no forecast highs returned")
            continue
        upcoming = sorted(highs.items())[:4]
        rendered = ", ".join(f"{d}: {t:.0f}F" for d, t in upcoming)
        ok(f"{series} ({station.name}): {rendered}")

    warn(
        "Confirm each station above is the one its Kalshi series actually resolves "
        "on — the series rules name it. A wrong station biases the model silently."
    )


def check_end_to_end() -> None:
    print("\n=== End to end ===")
    contracts = kalshi.fetch_weather_markets()
    if not contracts:
        bad("fetch_weather_markets returned nothing")
        return
    ok(f"{len(contracts)} contracts within {settings.event_days_ahead} days")

    forecasts = weather_forecast.get_forecast_highs(contracts)
    if not forecasts:
        bad("No forecasts matched any event")
        return
    ok(f"{len(forecasts)} events have a forecast")

    candidates = build_candidates(contracts, forecasts, now=datetime.utcnow())
    if not candidates:
        bad("No candidates priced")
        return
    ok(f"{len(candidates)} contracts priced")

    from src.analysis.event_screener import screen_event_candidates

    passed = screen_event_candidates(candidates)
    print(f"\n  Top edges (screener kept {len(passed)}):")
    for candidate in sorted(candidates, key=lambda c: c.edge, reverse=True)[:8]:
        print(
            f"    {candidate.ticker:<32} fair {candidate.fair_prob:.3f} "
            f"ask {candidate.market_prob:.3f} edge {candidate.edge:+.3f}"
        )
    if not passed:
        warn(
            "Nothing passed the screener. That is the expected result most scans — "
            "it is only a problem if it never passes anything over several days."
        )


def main() -> int:
    print(f"DeepSwing event source check — {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    host, markets = check_kalshi()
    if host:
        check_market_schema(markets)
    check_nws()
    if host:
        check_end_to_end()
    print("\nDone. Resolve every FAIL above before enabling event_dry_run=False.")
    return 0 if host else 1


if __name__ == "__main__":
    raise SystemExit(main())
