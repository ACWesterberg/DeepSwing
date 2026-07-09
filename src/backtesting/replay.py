from __future__ import annotations

import json
import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from config.settings import BASE_DIR, settings
from src.analysis.regime import classify_regime
from src.analysis.screener import screen_candidates
from src.analysis.technical import compute_signals
from src.backtesting.history import WARMUP_CALENDAR_DAYS, load_ohlcv_history, slice_asof
from src.backtesting.macro_history import MacroHistory
from src.backtesting.news_history import (
    fetch_market_headlines_asof,
    fetch_ticker_news_asof,
    format_headlines_block,
)
from src.data.watchlist import get_omxs30_tickers, get_us_tickers

logger = logging.getLogger(__name__)

NewsMode = Literal["raw", "analyzed", "off"]

DEFAULT_TRAINSET_PATH = BASE_DIR / "data" / "backtest" / "trainset.jsonl"

# Fewer forward bars than this and the outcome label would mostly be an
# arbitrary early timeout — skip the example instead.
_MIN_FORWARD_BARS = 5

_MIN_HISTORY_ROWS = 210

_INSIDER_LINE = "Insider activity: no data available (historical replay)."
_HEURISTICS_COLD_START = "No relevant heuristics yet."


@dataclass
class ReplayConfig:
    start: date
    end: date
    markets: list[str] = field(default_factory=lambda: ["nordic", "us"])
    stride_days: int = 2
    ticker_cooldown_days: int = 10
    max_hold_days: int = 20
    target_rrr: float = 2.5
    news_mode: NewsMode = "raw"
    out_path: Path = DEFAULT_TRAINSET_PATH
    max_examples: Optional[int] = None


@dataclass
class SimulatedOutcome:
    entry_date: date
    entry_price: float
    stop_loss: float
    target: float
    exit_date: date
    exit_price: float
    exit_reason: str  # "stop_loss" | "take_profit" | "timeout"
    days_held: int
    gross_pnl_pct: float
    net_pnl_pct: float


def simulate_outcome(
    df: pd.DataFrame,
    scan_day: date,
    atr: float,
    market: str,
    target_rrr: float,
    max_hold_days: int,
) -> Optional[SimulatedOutcome]:
    """What a disciplined BUY decided at `scan_day` close would have realized:
    entry at next open, ATR stop, RRR target, timeout exit. If stop and target
    are both touched in one bar the stop is assumed to fill first (conservative)."""
    fwd = df[df.index.date > scan_day]
    if len(fwd) < _MIN_FORWARD_BARS or atr <= 0:
        return None

    entry = float(fwd.iloc[0]["Open"])
    if entry <= 0:
        return None
    stop = entry - settings.atr_stop_multiplier * atr
    if stop <= 0:
        return None
    target = entry + target_rrr * (entry - stop)

    window = fwd.iloc[:max_hold_days]
    exit_price: Optional[float] = None
    exit_reason = ""
    exit_date: Optional[date] = None

    for ts, bar in window.iterrows():
        bar_open, bar_high, bar_low = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
        if bar_open <= stop:
            exit_price, exit_reason = bar_open, "stop_loss"  # gapped through the stop
        elif bar_low <= stop:
            exit_price, exit_reason = stop, "stop_loss"
        elif bar_open >= target:
            exit_price, exit_reason = bar_open, "take_profit"
        elif bar_high >= target:
            exit_price, exit_reason = target, "take_profit"
        if exit_reason:
            exit_date = ts.date()
            break

    if exit_price is None or exit_date is None:
        last = window.iloc[-1]
        exit_price = float(last["Close"])
        exit_reason = "timeout"
        exit_date = window.index[-1].date()

    gross = (exit_price - entry) / entry
    round_trip_cost = 2 * settings.simulated_slippage + 2 * settings.commission_pct
    if market == "us":
        round_trip_cost += 2 * settings.fx_commission_pct

    entry_date = fwd.index[0].date()
    return SimulatedOutcome(
        entry_date=entry_date,
        entry_price=entry,
        stop_loss=stop,
        target=target,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        days_held=(exit_date - entry_date).days,
        gross_pnl_pct=gross,
        net_pnl_pct=gross - round_trip_cost,
    )


def generate_trainset(cfg: ReplayConfig) -> dict:
    """Replay history and write one JSONL training example per screened candidate,
    with prompt inputs built exactly as DecisionEngine.decide() builds them live."""
    seen_keys = _load_seen_keys(cfg.out_path)
    if seen_keys:
        logger.info("Resuming: %d examples already in %s", len(seen_keys), cfg.out_path)

    stats = {
        "examples": 0, "skipped_seen": 0, "skipped_cooldown": 0,
        "skipped_no_outcome": 0, "vix_halt_days": 0, "scan_days": 0,
        "winners": 0, "losers": 0,
    }

    load_start = cfg.start - timedelta(days=WARMUP_CALENDAR_DAYS)
    load_end = min(cfg.end + timedelta(days=cfg.max_hold_days * 2), date.today())
    macro = MacroHistory(load_start, load_end)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_path, "a", encoding="utf-8") as out:
        for market in cfg.markets:
            _generate_market(market, cfg, macro, load_start, load_end, out, seen_keys, stats)
            if cfg.max_examples and stats["examples"] >= cfg.max_examples:
                break

    logger.info(
        "Trainset generation done: %d examples (%d winners / %d losers) → %s",
        stats["examples"], stats["winners"], stats["losers"], cfg.out_path,
    )
    return stats


def _generate_market(
    market: str,
    cfg: ReplayConfig,
    macro: MacroHistory,
    load_start: date,
    load_end: date,
    out,
    seen_keys: set[tuple[str, str, str]],
    stats: dict,
) -> None:
    watchlist = get_omxs30_tickers() if market == "nordic" else get_us_tickers()
    ohlcv = load_ohlcv_history(watchlist, market, load_start, load_end)
    if not ohlcv:
        logger.warning("No OHLCV history for %s — skipping market", market)
        return

    trading_days = sorted({d for df in ohlcv.values() for d in df.index.date})
    scan_days = [d for d in trading_days if cfg.start <= d <= cfg.end][:: cfg.stride_days]
    last_sampled: dict[str, date] = {}

    for day in scan_days:
        if cfg.max_examples and stats["examples"] >= cfg.max_examples:
            return
        stats["scan_days"] += 1

        vix = macro.vix(day)
        if vix is not None and vix >= settings.vix_halt_threshold:
            stats["vix_halt_days"] += 1
            continue

        analysis_map: dict[str, tuple] = {}
        for ticker, df in ohlcv.items():
            sliced = slice_asof(df, day)
            if len(sliced) < _MIN_HISTORY_ROWS:
                continue
            signals = compute_signals(ticker, sliced)
            if signals is None:
                continue
            analysis_map[ticker] = (signals, classify_regime(sliced))

        candidates = screen_candidates(analysis_map, market)
        if not candidates:
            continue

        macro_context = macro.context(market, day)
        if cfg.news_mode != "off":
            market_env = format_headlines_block(
                fetch_market_headlines_asof(market, day, limit=settings.market_news_max_headlines),
                day,
                header="Recent market-wide headlines (newest first):",
            )
            macro_context = f"{macro_context}\n\n{market_env}"

        for candidate in candidates:
            key = (market, candidate.ticker, day.isoformat())
            if key in seen_keys:
                # Still arms the cooldown so a resumed run reproduces the
                # original sampling pattern instead of back-filling skipped days
                last_sampled[candidate.ticker] = day
                stats["skipped_seen"] += 1
                continue
            last = last_sampled.get(candidate.ticker)
            if last and _trading_days_between(trading_days, last, day) < cfg.ticker_cooldown_days:
                stats["skipped_cooldown"] += 1
                continue

            outcome = simulate_outcome(
                ohlcv[candidate.ticker], day, candidate.signals.atr_14,
                market, cfg.target_rrr, cfg.max_hold_days,
            )
            if outcome is None:
                stats["skipped_no_outcome"] += 1
                continue

            news_summary = _build_news_summary(candidate, market, day, cfg.news_mode)
            record = {
                "schema": 1,
                "market": market,
                "ticker": candidate.ticker,
                "scan_date": day.isoformat(),
                "entry_date": outcome.entry_date.isoformat(),
                "exit_date": outcome.exit_date.isoformat(),
                "exit_reason": outcome.exit_reason,
                "entry_price": round(outcome.entry_price, 4),
                "exit_price": round(outcome.exit_price, 4),
                "stop_loss": round(outcome.stop_loss, 4),
                "target": round(outcome.target, 4),
                "days_held": outcome.days_held,
                "pnl_pct": round(outcome.net_pnl_pct, 6),
                "gross_pnl_pct": round(outcome.gross_pnl_pct, 6),
                "news_mode": cfg.news_mode,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "inputs": {
                    "technicals": candidate.signals.to_prompt_str(),
                    "regime": candidate.regime.to_prompt_str(),
                    "news_summary": news_summary,
                    "macro_context": macro_context,
                    "heuristics": _HEURISTICS_COLD_START,
                },
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            seen_keys.add(key)
            last_sampled[candidate.ticker] = day
            stats["examples"] += 1
            stats["winners" if outcome.net_pnl_pct > 0 else "losers"] += 1

            if cfg.max_examples and stats["examples"] >= cfg.max_examples:
                return


def _build_news_summary(candidate, market: str, day: date, news_mode: NewsMode) -> str:
    if news_mode == "off":
        return f"No recent relevant news found.\n{_INSIDER_LINE}"

    articles = fetch_ticker_news_asof(candidate.ticker, market, day)
    if news_mode == "analyzed":
        from src.agent.news_analyzer import analyze_news

        summary = analyze_news(
            ticker=candidate.ticker,
            market=market,
            current_price=candidate.signals.current_price,
            technicals_brief=(
                f"Price {candidate.signals.current_price:.4f}, RSI {candidate.signals.rsi_14:.1f}"
            ),
            articles=articles,
        )
    else:
        summary = format_headlines_block(articles, day)
    return f"{summary}\n{_INSIDER_LINE}"


def _trading_days_between(trading_days: list[date], last: date, day: date) -> int:
    return bisect_right(trading_days, day) - bisect_left(trading_days, last) - 1


def _load_seen_keys(path: Path) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    if not path.exists():
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r["market"], r["ticker"], r["scan_date"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return keys
