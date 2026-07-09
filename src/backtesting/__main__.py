from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

from src.backtesting.replay import DEFAULT_TRAINSET_PATH, ReplayConfig, generate_trainset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.backtesting",
        description="Historical replay: generate MIPRO training examples from past data "
        "and optimize prompts offline (see src/backtesting/README.md).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Replay history and write a trainset JSONL")
    gen.add_argument("--start", type=date.fromisoformat, required=True)
    gen.add_argument("--end", type=date.fromisoformat, required=True)
    gen.add_argument("--markets", nargs="+", choices=["nordic", "us"], default=["nordic", "us"])
    gen.add_argument("--stride", type=int, default=2, help="Scan every Nth trading day")
    gen.add_argument("--cooldown", type=int, default=10, help="Trading days before re-sampling a ticker")
    gen.add_argument("--max-hold", type=int, default=20, help="Timeout exit after N trading days")
    gen.add_argument("--target-rrr", type=float, default=2.5)
    gen.add_argument("--news-mode", choices=["raw", "analyzed", "off"], default="raw")
    gen.add_argument("--out", type=Path, default=DEFAULT_TRAINSET_PATH)
    gen.add_argument("--max-examples", type=int, default=None)

    opt = sub.add_parser("optimize", help="Run MIPROv2 on a generated trainset")
    opt.add_argument("--track", choices=["claude", "gpt"], required=True)
    opt.add_argument("--dataset", type=Path, default=DEFAULT_TRAINSET_PATH)
    opt.add_argument("--threads", type=int, default=1)

    st = sub.add_parser("stats", help="Show trainset statistics")
    st.add_argument("--dataset", type=Path, default=DEFAULT_TRAINSET_PATH)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "generate":
        if args.end <= args.start:
            parser.error("--end must be after --start")
        cfg = ReplayConfig(
            start=args.start,
            end=args.end,
            markets=args.markets,
            stride_days=args.stride,
            ticker_cooldown_days=args.cooldown,
            max_hold_days=args.max_hold,
            target_rrr=args.target_rrr,
            news_mode=args.news_mode,
            out_path=args.out,
            max_examples=args.max_examples,
        )
        stats = generate_trainset(cfg)
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "optimize":
        from src.backtesting.optimize import run_backtest_mipro

        out = run_backtest_mipro(args.track, args.dataset, num_threads=args.threads)
        return 0 if out else 1

    if args.command == "stats":
        from src.backtesting.optimize import dataset_stats, load_records

        print(json.dumps(dataset_stats(load_records(args.dataset)), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
