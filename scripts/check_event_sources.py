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
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from config.settings import settings  # noqa: E402
from src.analysis.event_model import (  # noqa: E402
    bucket_bounds, build_candidates, forecast_sigma,
)
from src.data import kalshi, weather_forecast  # noqa: E402

KALSHI_HOSTS = [
    settings.kalshi_api_base,
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://external-api.kalshi.com/trade-api/v2",
]

# Fields the parser needs on every market, each as the list of names it accepts —
# Kalshi serves prices as either `*_dollars` decimal strings or bare integer cents.
REQUIRED_FIELDS = {
    "ticker": ["ticker"],
    "event ticker": ["event_ticker"],
    "close time": ["close_time"],
    "strike type": ["strike_type"],
    "yes bid": ["yes_bid_dollars", "yes_bid"],
    "yes ask": ["yes_ask_dollars", "yes_ask"],
    "open interest": ["open_interest", "open_interest_fp"],
}


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
    print(f"  Keys on a sample market:\n    {sorted(sample)}")

    for label, names in REQUIRED_FIELDS.items():
        if not any(n in sample for n in names):
            bad(f"No field for {label} — parser looks for {names}")
            warn("Update _parse_market in src/data/kalshi.py to the real name above.")
    # floor/cap are legitimately one-sided: a `greater` market has no cap_strike.
    if "floor_strike" not in sample and "cap_strike" not in sample:
        bad("Market has neither floor_strike nor cap_strike")

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

    # The check that matters: a renamed price field parses as 0.00 rather than
    # raising, so every market silently becomes "unquoted" and nothing trades.
    quoted = [c for c in usable if c.yes_ask > 0]
    if quoted:
        ok(f"{len(quoted)}/{len(usable)} sampled markets have a non-zero ask")
    else:
        bad("Every sampled ask parsed as 0.00 — the price field names have changed")
        warn("Compare the key list above against _price() in src/data/kalshi.py")

    # Show a `between` market too — its raw floor/cap reveal whether Kalshi quotes
    # whole degrees (needing the continuity correction) or half-degree boundaries.
    shown = usable[:3]
    between = next((c for c in usable if c.strike_type == "between"), None)
    if between is not None and between not in shown:
        shown.append(between)
    for contract in shown:
        try:
            lo, hi = bucket_bounds(contract)
            print(
                f"    {contract.ticker}: bid {contract.yes_bid:.2f} ask {contract.yes_ask:.2f} "
                f"oi {contract.open_interest} | {contract.strike_type} "
                f"floor={contract.floor_strike} cap={contract.cap_strike} "
                f"-> YES on [{lo}, {hi}] degF"
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



def check_distribution() -> None:
    """
    Compare the model's distribution against the market's for one event.

    The buckets of an event are exhaustive, so the market's own prices are a
    probability distribution over the daily high. Setting it beside ours says
    which knob is wrong: a shifted centre means the forecast (or the station it
    comes from) is off, while a matching centre with a wider spread means sigma
    is too large. A blanket "edge" on every bucket below the forecast is the
    signature of one of those, not of a mispriced market.
    """
    print("\n=== Model vs market distribution ===")
    contracts = kalshi.fetch_weather_markets()
    forecasts = weather_forecast.get_forecast_highs(contracts)
    if not contracts or not forecasts:
        warn("Nothing to compare")
        return

    by_event: dict[str, list] = {}
    for c in contracts:
        if c.event_ticker in forecasts:
            by_event.setdefault(c.event_ticker, []).append(c)
    if not by_event:
        warn("No event has both markets and a forecast")
        return

    event = max(by_event, key=lambda e: len(by_event[e]))
    group = sorted(by_event[event], key=lambda c: (c.floor_strike or -999))
    forecast = forecasts[event]
    print(f"  {event} — NWS forecast high {forecast:.1f}F, {len(group)} buckets")

    candidates = build_candidates(group, forecasts, now=datetime.utcnow())
    by_ticker = {c.ticker: c for c in candidates}

    print(f"    {'bucket':<26} {'model':>7} {'market':>7} {'diff':>7}")
    weighted, total, mids = 0.0, 0.0, []
    for contract in group:
        cand = by_ticker.get(contract.ticker)
        if cand is None:
            continue
        market = contract.mid
        label = f"{contract.strike_type} {contract.floor_strike}-{contract.cap_strike}"
        print(f"    {label:<26} {cand.fair_prob:>7.3f} {market:>7.3f} "
              f"{cand.fair_prob - market:>+7.3f}")
        # Only closed buckets have a defined midpoint for the moment estimate.
        if contract.floor_strike is not None and contract.cap_strike is not None:
            mid_temp = (contract.floor_strike + contract.cap_strike) / 2
            weighted += mid_temp * market
            total += market
            mids.append((mid_temp, market))

    if total <= 0:
        warn("No two-sided closed buckets — cannot infer the market's distribution")
        return

    market_mean = weighted / total
    market_sd = math.sqrt(sum(p * (m - market_mean) ** 2 for m, p in mids) / total)
    print(f"\n  Market implies mean {market_mean:.1f}F, sd {market_sd:.2f}F")
    print(f"  Model uses      mean {forecast:.1f}F, sd {forecast_sigma(1.0):.2f}F "
          f"(forecast_sigma at 1 day)")

    shift = market_mean - forecast
    if abs(shift) >= 1.0:
        bad(f"Centre is off by {shift:+.1f}F — the forecast or the station is wrong, "
            f"and every bucket is mispriced as a result")
    else:
        ok(f"Centre agrees within {abs(shift):.1f}F")

    if market_sd > 0 and forecast_sigma(1.0) > market_sd * 1.5:
        bad(f"forecast_sigma is {forecast_sigma(1.0) / market_sd:.1f}x the market's "
            f"spread — every bucket away from the centre looks underpriced")
        warn("Lower FORECAST_SIGMA_DAY1 toward the market's sd before trading")
    else:
        ok("Spread is in the same range as the market's")


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

    quoted = [c for c in candidates if c.market_prob > 0]
    if quoted:
        ok(f"{len(quoted)}/{len(candidates)} have a tradeable ask")
    else:
        bad("No contract has a tradeable ask — edges below are meaningless")

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


def print_rules(markets: list[dict]) -> None:
    """Kalshi's own resolution rules — they name the station the series settles on."""
    print("\n=== Resolution rules (confirm the station) ===")
    rules = (markets[0].get("rules_primary") or "").strip() if markets else ""
    print(f"  {rules[:600] or 'no rules_primary on the response'}")


def main() -> int:
    print(f"DeepSwing event source check — {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    host, markets = check_kalshi()
    if host:
        check_market_schema(markets)
    check_nws()
    if host:
        check_end_to_end()
        check_distribution()
        print_rules(markets)
    print("\nDone. Resolve every FAIL above before enabling event_dry_run=False.")
    return 0 if host else 1


if __name__ == "__main__":
    raise SystemExit(main())
