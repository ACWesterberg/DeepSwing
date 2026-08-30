"""
Offline evaluation of the decision program against persisted decisions.

Every prompt change previously cost a month: the backtester contains no model
(it buys every screener survivor), and `metrics_by_program` can only compare
programs that already ran sequentially against different markets. So the only
way to ask "is this prompt better" was to trade it.

Persisted PASS/BLOCKED decisions already carry everything needed to ask offline.
`Decision.entry_inputs` holds the exact five DSPy fields the program was called
with, `price` and `atr` pin the decision-time state, and `_label_forward_path`
derives the ground-truth R from what the underlying then did — mirroring the
live exit policy, breakeven floor included. Replaying is the same
`program(**entry_inputs)` call the live path makes.

Two phases, deliberately separate: building the corpus is network-bound (one
OHLCV fetch per ticker) and is cached to disk; scoring is LLM-bound and reads
the cache, so comparing programs costs no further price data.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Optional

from config.settings import settings
from src.portfolio.metrics import decision_metric

logger = logging.getLogger(__name__)


@dataclass
class ReplayExample:
    """One decision, plus what the market went on to do."""
    track: str
    ticker: str
    market: str
    timestamp: str
    entry_inputs: dict
    label: str          # BUY (setup paid) | PASS (it didn't)
    r_multiple: float   # ground-truth R, the unit the live metric uses


@dataclass
class ReplayResult:
    program: str
    n: int
    mean_metric: float
    buys: int
    mean_r_taken: float      # R actually captured by the BUYs it made
    total_r_taken: float
    missed_buys: int         # label BUY, program passed
    false_buys: int          # label PASS, program bought
    recall: float            # of the setups that paid, how many were taken
    precision: float         # of the setups taken, how many paid

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.program:<24} n={self.n:<5} metric={self.mean_metric:.4f}  "
            f"buys={self.buys:<4} meanR={self.mean_r_taken:+.2f}  "
            f"totalR={self.total_r_taken:+7.1f}  "
            f"precision={self.precision:.0%} recall={self.recall:.0%}"
        )


# --- corpus construction -----------------------------------------------------


def _session_for(db_path: Optional[Path]):
    """A session bound to an arbitrary DB file, so a backup can be replayed."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.db import get_session

    if db_path is None:
        return get_session()
    engine = create_engine(
        f"sqlite:///{Path(db_path).expanduser()}",
        connect_args={"check_same_thread": False},
    )
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)()


def _select_rows(
    rows: list[dict],
    limit: int,
    max_per_ticker: int,
    max_tickers: int,
) -> list[dict]:
    """
    Pick a diverse subset: most-decided tickers first, but bounded per ticker.

    `_persist_decisions` keeps one blob per (track, ticker) per day, so a name
    decided daily for six weeks contributes ~45 rows per track. Those rows are
    highly correlated — the same setup on consecutive days — so letting one
    ticker dominate inflates `n` without adding independent evidence, and makes
    a prompt look more precisely measured than it is. Frequency ordering keeps
    the fetch bounded; the per-ticker cap keeps the sample broad.
    """
    by_ticker: dict[str, list[dict]] = {}
    for row in rows:
        by_ticker.setdefault(row["ticker"], []).append(row)

    ranked = sorted(by_ticker.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected: list[dict] = []
    for ticker, ticker_rows in ranked[:max_tickers]:
        selected.extend(ticker_rows[:max_per_ticker])
        if len(selected) >= limit:
            break
    return selected[:limit]


def _batch_prices(rows: list[dict]) -> dict[str, object]:
    """
    One chunked download per market rather than one call per ticker.

    The per-ticker path routes Nordic through Alpha Vantage, whose free tier
    allows 25 requests a day — a few hundred tickers exhausts it and every
    subsequent Nordic fetch fails. These are the same batch helpers the scan
    loop uses to pull the whole Nordic universe, and they don't touch it.
    """
    from src.data.market_data import (
        fetch_batch_eu, fetch_batch_nordic, fetch_batch_us,
    )

    fetchers = {"nordic": fetch_batch_nordic, "eu": fetch_batch_eu, "us": fetch_batch_us}
    by_market: dict[str, set[str]] = {}
    for row in rows:
        by_market.setdefault(row["market"], set()).add(row["ticker"])

    prices: dict[str, object] = {}
    for market, tickers in by_market.items():
        fetch = fetchers.get(market)
        if fetch is None:
            logger.warning("No batch fetcher for market %r — skipping %d tickers",
                           market, len(tickers))
            continue
        logger.info("Fetching %d %s tickers...", len(tickers), market)
        try:
            got = fetch(sorted(tickers))
        except Exception as exc:
            logger.warning("Batch fetch failed for %s: %s", market, exc)
            continue
        logger.info("  %s: %d/%d returned", market, len(got), len(tickers))
        prices.update(got)
    return prices


def build_corpus(
    track: Optional[str] = None,
    db_path: Optional[Path] = None,
    limit: int = 500,
    horizon_days: Optional[int] = None,
    market: Optional[str] = None,
    max_per_ticker: int = 5,
    max_tickers: int = 150,
    on_progress: Optional[Callable[[list["ReplayExample"]], None]] = None,
) -> list[ReplayExample]:
    """Label aged PASS/BLOCKED decisions from what the price actually did."""
    # Local: labelling needs the live pipeline, but scoring must not — keeping
    # these out of module scope is what lets the reference predictors below run
    # without dspy installed, so the harness can be validated with no model.
    from src.db import Decision
    from src.scheduler.optimizer import _label_forward_path

    horizon = timedelta(days=horizon_days or settings.counterfactual_horizon_days)
    cutoff = datetime.utcnow() - horizon

    session = _session_for(db_path)
    try:
        q = session.query(Decision).filter(
            Decision.action.in_(("PASS", "BLOCKED")),
            Decision.entry_inputs.isnot(None),
            Decision.price.isnot(None),
            Decision.timestamp <= cutoff,
        )
        if track:
            q = q.filter(Decision.track == track)
        if market:
            q = q.filter(Decision.market == market)
        rows = [
            {
                "track": r.track, "ticker": r.ticker, "market": r.market,
                "price": r.price, "atr": r.atr, "timestamp": r.timestamp,
                "entry_inputs": r.entry_inputs,
            }
            for r in q.order_by(Decision.timestamp.desc()).all()
        ]
    finally:
        session.close()

    if not rows:
        return []

    selected = _select_rows(rows, limit, max_per_ticker, max_tickers)
    logger.info("Selected %d decisions across %d tickers (from %d rows)",
                len(selected), len({r["ticker"] for r in selected}), len(rows))

    prices = _batch_prices(selected)

    corpus: list[ReplayExample] = []
    try:
        for row in selected:
            df = prices.get(row["ticker"])
            if df is None or getattr(df, "empty", True):
                continue

            start = row["timestamp"].date()
            end = (row["timestamp"] + horizon).date()
            window = df[(df.index.date > start) & (df.index.date <= end)]
            if len(window) < 3 or window["Close"].dropna().empty:
                continue

            labelled = _label_forward_path(window, row["price"], row["atr"])
            if labelled is None:
                continue  # ambiguous drift, or no risk denominator
            label, r = labelled
            corpus.append(ReplayExample(
                track=row["track"], ticker=row["ticker"], market=row["market"],
                timestamp=row["timestamp"].isoformat(),
                entry_inputs=row["entry_inputs"], label=label, r_multiple=r,
            ))
            if on_progress and len(corpus) % 50 == 0:
                on_progress(corpus)
    except KeyboardInterrupt:
        # Labelling a few hundred rows takes a while; keep what was earned.
        logger.warning("Interrupted — keeping %d labelled examples", len(corpus))

    logger.info("Replay corpus: %d labelled examples from %d selected decisions",
                len(corpus), len(selected))
    return corpus


def save_corpus(corpus: Iterable[ReplayExample], path: Path) -> int:
    rows = [asdict(e) for e in corpus]
    Path(path).write_text(json.dumps(rows, indent=2))
    return len(rows)


def load_corpus(path: Path) -> list[ReplayExample]:
    return [ReplayExample(**r) for r in json.loads(Path(path).read_text())]


# --- scoring -----------------------------------------------------------------


class _Prediction:
    """Minimal stand-in for a DSPy prediction — the metric only reads .action."""

    def __init__(self, action: str):
        self.action = action


def score_program(
    corpus: list[ReplayExample],
    predict: Callable[[ReplayExample], str],
    name: str = "program",
) -> ReplayResult:
    """
    Run `predict` over the corpus and score it exactly as MIPRO would.

    Reports the metric *and* the realised R behind it. The metric alone hides
    the do-nothing degenerate: PASS scores exactly 0.5 on every example, so a
    program that never buys lands at 0.5 regardless of the corpus. Precision,
    recall and total R are what distinguish "correctly cautious" from "inert".
    """
    if not corpus:
        raise ValueError("empty corpus — build it first")

    total_metric = 0.0
    buys = missed = false_buys = 0
    r_taken: list[float] = []

    for ex in corpus:
        action = str(predict(ex)).upper()
        example = type("E", (), {"r_multiple": ex.r_multiple})()
        total_metric += decision_metric(example, _Prediction(action))

        if action == "BUY":
            buys += 1
            r_taken.append(ex.r_multiple)
            if ex.label != "BUY":
                false_buys += 1
        elif ex.label == "BUY":
            missed += 1

    payers = sum(1 for e in corpus if e.label == "BUY")
    return ReplayResult(
        program=name,
        n=len(corpus),
        mean_metric=total_metric / len(corpus),
        buys=buys,
        mean_r_taken=(sum(r_taken) / len(r_taken)) if r_taken else 0.0,
        total_r_taken=sum(r_taken),
        missed_buys=missed,
        false_buys=false_buys,
        recall=((payers - missed) / payers) if payers else 0.0,
        precision=((buys - false_buys) / buys) if buys else 0.0,
    )


# --- reference predictors ----------------------------------------------------
#
# These need no LLM, so the harness can be validated for free. If it cannot
# order oracle > always_buy > always_pass on a corpus with positive expectancy,
# it is not measuring anything and no verdict it gives about a real prompt
# should be believed.

def always_buy(_: ReplayExample) -> str:
    return "BUY"


def always_pass(_: ReplayExample) -> str:
    return "PASS"


def oracle(example: ReplayExample) -> str:
    """Upper bound: takes exactly the setups that paid."""
    return example.label


def dspy_program(program) -> Callable[[ReplayExample], str]:
    """Wrap a compiled or baseline DSPy program as a predictor."""
    def _predict(example: ReplayExample) -> str:
        try:
            result = program(**example.entry_inputs)
            action = str(result.action).upper()
            return action if action in ("BUY", "PASS") else "PASS"
        except Exception as exc:
            logger.warning("Replay call failed for %s: %s", example.ticker, exc)
            return "PASS"
    return _predict
