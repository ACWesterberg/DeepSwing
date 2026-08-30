"""
Compare decision programs offline, against decisions the system already made.

Answers "is this prompt better than that one" in minutes over hundreds of
examples, instead of a month over thirty. The backtester cannot do this — it
contains no model at all and buys every screener survivor — and
`metrics_by_program` can only compare programs that already ran sequentially
against different markets.

Two phases, because they bottleneck on different things:

    # once: label decisions from what the price did (network-bound, cached)
    python3 scripts/replay_decisions.py build --db ~/deepswing_prereset/deepswing_*.db

    # free: validate the harness itself, no LLM calls
    python3 scripts/replay_decisions.py score --reference

    # then: score real programs (LLM-bound)
    python3 scripts/replay_decisions.py score --program baseline
    python3 scripts/replay_decisions.py score --program compiled/claude_trade_decision.json

Read `--reference` output first. It runs oracle / always-buy / always-pass,
which need no model. If those three don't order sensibly, the harness is broken
and nothing it says about a real prompt means anything.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.replay import (  # noqa: E402
    always_buy, always_pass, build_corpus, dspy_program, load_corpus,
    oracle, save_corpus, score_program,
)

_DEFAULT_CACHE = Path("data/replay_corpus.json")


def _build(args) -> int:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _checkpoint(partial):
        # Labelling a few hundred rows takes a while; a Ctrl-C should not
        # discard everything already earned.
        save_corpus(partial, out)
        print(f"  ... {len(partial)} labelled (checkpointed)")

    corpus = build_corpus(
        track=args.track,
        db_path=Path(args.db) if args.db else None,
        limit=args.limit,
        horizon_days=args.horizon,
        market=args.market,
        max_per_ticker=args.max_per_ticker,
        max_tickers=args.max_tickers,
        on_progress=_checkpoint,
    )
    if not corpus:
        print("No labellable decisions found.\n"
              "Decisions need to be older than the counterfactual horizon "
              "(default 14 days) and carry entry_inputs + price + atr.")
        return 1

    n = save_corpus(corpus, out)
    payers = sum(1 for e in corpus if e.label == "BUY")
    tickers = len({e.ticker for e in corpus})
    print(f"Wrote {n} labelled examples to {args.out}")
    print(f"  {tickers} distinct tickers (one batch fetch per market, not per ticker)")
    print(f"  {payers} setups paid ({payers/n:.0%}), {n - payers} did not")
    print(f"  mean R if every one were taken: {sum(e.r_multiple for e in corpus)/n:+.2f}")
    return 0


def _score(args) -> int:
    path = Path(args.corpus)
    if not path.exists():
        print(f"No corpus at {path} — run `build` first.")
        return 1
    corpus = load_corpus(path)

    results = []
    if args.reference:
        # No LLM calls: this is the harness validating itself.
        results += [
            score_program(corpus, oracle, "oracle (upper bound)"),
            score_program(corpus, always_buy, "always-buy"),
            score_program(corpus, always_pass, "always-pass"),
        ]
    for spec in args.program or []:
        results.append(score_program(corpus, dspy_program(_load_program(spec)), spec))

    if not results:
        print("Nothing to score — pass --reference and/or --program.")
        return 1

    print(f"\n{len(corpus)} examples from {args.corpus}\n")
    for r in results:
        print("  " + r.summary())
    print("\nmetric is what MIPRO optimises; always-pass scores exactly 0.500 on any")
    print("corpus, so read totalR and precision/recall to tell caution from inertia.")
    return 0


def _load_program(spec: str):
    import dspy

    from src.agent.decision import TradeDecision

    program = dspy.Predict(TradeDecision)
    if spec != "baseline":
        program.load(spec)
    return program


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="label decisions from subsequent price action")
    b.add_argument("--db", help="DB to read (default: the live one). Point at a "
                                "backup to use decisions a reset deleted.")
    b.add_argument("--track", help="claude | gpt (default: both)")
    b.add_argument("--limit", type=int, default=500)
    b.add_argument("--market", help="nordic | eu | us (default: all). US/EU are "
                                    "pure yfinance if Nordic data is unavailable.")
    b.add_argument("--max-per-ticker", type=int, default=5, dest="max_per_ticker",
                   help="cap rows from any one ticker — consecutive days on the "
                        "same name are correlated and inflate n (default 5)")
    b.add_argument("--max-tickers", type=int, default=150, dest="max_tickers")
    b.add_argument("--horizon", type=int, help="forward days (default: settings)")
    b.add_argument("--out", default=str(_DEFAULT_CACHE))
    b.set_defaults(func=_build)

    s = sub.add_parser("score", help="run programs over the cached corpus")
    s.add_argument("--corpus", default=str(_DEFAULT_CACHE))
    s.add_argument("--reference", action="store_true",
                   help="score oracle/always-buy/always-pass — no LLM calls")
    s.add_argument("--program", action="append",
                   help="'baseline' or a path to a compiled program JSON; repeatable")
    s.set_defaults(func=_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
