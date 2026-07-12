from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal, Optional

from config.settings import settings
from src.analysis.options_math import bs_delta, bs_theta_per_day

logger = logging.getLogger(__name__)

RightType = Literal["call", "put"]

CONTRACT_MULTIPLIER = 100


@dataclass
class OptionContract:
    contract_symbol: str
    underlying: str
    right: RightType
    strike: float          # USD
    expiry: date
    bid: float             # USD per share
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    dte: int
    mid: float
    spread_pct: float
    delta: float
    theta_per_day: float

    def display_name(self) -> str:
        return f"{self.underlying} {self.right[0].upper()}${self.strike:g} {self.expiry.isoformat()}"

    def to_prompt_line(self, index: int) -> str:
        return (
            f"[{index}] {self.display_name()} | DTE {self.dte} | mid ${self.mid:.2f} "
            f"(bid {self.bid:.2f}/ask {self.ask:.2f}, spread {self.spread_pct*100:.1f}%) | "
            f"delta {self.delta:.2f} | theta {self.theta_per_day:.3f}/day | "
            f"IV {self.implied_vol*100:.1f}% | OI {self.open_interest} | vol {self.volume}"
        )


def fetch_chain_shortlist(ticker: str, spot: float, right: RightType = "call") -> list[OptionContract]:
    """Fetch the chain and return the liquid, swing-suitable contracts (<= shortlist_size)."""
    import yfinance as yf

    try:
        yft = yf.Ticker(ticker)
        expiries = _expiries_in_window(yft.options)
    except Exception as exc:
        logger.warning("Options chain fetch failed for %s: %s", ticker, exc)
        return []
    if not expiries:
        logger.debug("No expiries in %d-%d DTE window for %s", settings.options_min_dte, settings.options_max_dte, ticker)
        return []

    contracts: list[OptionContract] = []
    for expiry in expiries:
        try:
            chain = yft.option_chain(expiry.isoformat())
        except Exception as exc:
            logger.warning("option_chain(%s) failed for %s: %s", expiry, ticker, exc)
            continue
        frame = chain.calls if right == "call" else chain.puts
        contracts.extend(_parse_rows(frame, ticker, right, expiry, spot))

    contracts = [c for c in contracts if _passes_filters(c)]
    contracts.sort(key=lambda c: c.open_interest, reverse=True)
    return contracts[: settings.options_shortlist_size]


def fetch_contract_quotes(underlying: str, expiry: date, right: RightType, symbols: set[str]) -> dict[str, float]:
    """Current mid quotes (USD/share) for the given contract symbols on one expiry."""
    import yfinance as yf

    try:
        chain = yf.Ticker(underlying).option_chain(expiry.isoformat())
    except Exception as exc:
        logger.warning("Quote refresh failed for %s %s: %s", underlying, expiry, exc)
        return {}
    frame = chain.calls if right == "call" else chain.puts

    quotes: dict[str, float] = {}
    for row in frame.itertuples():
        symbol = str(getattr(row, "contractSymbol", ""))
        if symbol not in symbols:
            continue
        bid = _f(getattr(row, "bid", 0))
        ask = _f(getattr(row, "ask", 0))
        last = _f(getattr(row, "lastPrice", 0))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
        if mid > 0:
            quotes[symbol] = mid
    return quotes


def format_shortlist(contracts: list[OptionContract]) -> str:
    return "\n".join(c.to_prompt_line(i) for i, c in enumerate(contracts))


def _expiries_in_window(expiry_strs: tuple[str, ...] | list[str]) -> list[date]:
    today = date.today()
    in_window = []
    for s in expiry_strs or []:
        try:
            expiry = datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (expiry - today).days
        if settings.options_min_dte <= dte <= settings.options_max_dte:
            in_window.append(expiry)
    return sorted(in_window)[: settings.options_expiries_considered]


def _parse_rows(frame, ticker: str, right: RightType, expiry: date, spot: float) -> list[OptionContract]:
    dte = (expiry - date.today()).days
    out: list[OptionContract] = []
    for row in frame.itertuples():
        bid = _f(getattr(row, "bid", 0))
        ask = _f(getattr(row, "ask", 0))
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        iv = _f(getattr(row, "impliedVolatility", 0))
        strike = _f(getattr(row, "strike", 0))
        if strike <= 0 or mid <= 0:
            continue
        out.append(OptionContract(
            contract_symbol=str(getattr(row, "contractSymbol", "")),
            underlying=ticker,
            right=right,
            strike=strike,
            expiry=expiry,
            bid=bid,
            ask=ask,
            last=_f(getattr(row, "lastPrice", 0)),
            volume=_i(getattr(row, "volume", 0)),
            open_interest=_i(getattr(row, "openInterest", 0)),
            implied_vol=iv,
            dte=dte,
            mid=mid,
            spread_pct=(ask - bid) / mid,
            delta=bs_delta(spot, strike, dte, iv, right, settings.options_risk_free_rate),
            theta_per_day=bs_theta_per_day(spot, strike, dte, iv, right, settings.options_risk_free_rate),
        ))
    return out


def _passes_filters(c: OptionContract) -> bool:
    return (
        settings.options_delta_min <= abs(c.delta) <= settings.options_delta_max
        and c.open_interest >= settings.options_min_open_interest
        and c.volume >= settings.options_min_volume
        and c.spread_pct <= settings.options_max_spread_pct
    )


def _f(value) -> float:
    try:
        f = float(value)
        return f if f == f else 0.0  # NaN guard
    except (TypeError, ValueError):
        return 0.0


def _i(value) -> int:
    try:
        i = float(value)
        return int(i) if i == i else 0
    except (TypeError, ValueError):
        return 0
