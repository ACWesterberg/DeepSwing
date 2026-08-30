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


def build_corpus(
    track: Optional[str] = None,
    db_path: Optional[Path] = None,
    limit: int = 500,
    horizon_days: Optional[int] = None,
) -> list[ReplayExample]:
    """Label aged PASS/BLOCKED decisions from what the price actually did."""
    # Local: labelling needs the live pipeline, but scoring must not — keeping
    # these out of module scope is what lets the reference predictors below run
    # without dspy installed, so the harness can be validated with no model.
    from src.data.market_data import fetch_ohlcv
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
        rows = [
            {
                "track": r.track, "ticker": r.ticker, "market": r.market,
                "price": r.price, "atr": r.atr, "timestamp": r.timestamp,
                "entry_inputs": r.entry_inputs,
            }
            for r in q.order_by(Decision.timestamp.desc()).limit(limit * 3).all()
        ]
    finally:
        session.close()

    corpus: list[ReplayExample] = []
    ohlcv_cache: dict[str, object] = {}
    for row in rows:
        if len(corpus) >= limit:
            break
        ticker = row["ticker"]
        if ticker not in ohlcv_cache:
            try:
                ohlcv_cache[ticker] = fetch_ohlcv(ticker, row["market"], period="1y")
            except Exception as exc:
                logger.debug("Replay price fetch failed for %s: %s", ticker, exc)
                ohlcv_cache[ticker] = None
        df = ohlcv_cache[ticker]
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
            track=row["track"], ticker=ticker, market=row["market"],
            timestamp=row["timestamp"].isoformat(),
            entry_inputs=row["entry_inputs"], label=label, r_multiple=r,
        ))

    logger.info("Replay corpus: %d labelled examples from %d decisions", len(corpus), len(rows))
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
