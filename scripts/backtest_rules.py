"""No-model backtest of the screener + exit rules over a universe, with presets.

Run on a machine with financedata (the Pi):
    venv/bin/python scripts/backtest_rules.py --market nordic --start 2023-01-01 --windows 3
    venv/bin/python scripts/backtest_rules.py --market us --preset legacy --set min_rrr=3.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

# Approximates the rules that traded Aug-Oct 2026 (2 wins in 35). The old
# momentum scoring no longer exists, so only the gates and exits are restored.
PRESETS: dict[str, dict] = {
    "current": {},
    "legacy": {
        "require_above_200sma": False, "rsi_min": 35.0, "rsi_max": 78.0,
        "volume_spike_multiplier": 1.2, "max_volume_ratio": 0.0,
        "min_atr_pct": 0.02, "max_atr_pct": 0.0,
        "hurst_on_returns": False, "regime_gate": True,
        "min_turnover_sek": 0.0, "min_price_sek": 0.0, "exclude_lse_international": False,
        "standard_stop_atr_multiplier": 1.0, "min_stop_atr_multiplier": 0.5,
        "atr_stop_multiplier": 1.5, "min_rrr": 2.5,
        "trailing_stop_atr_multiplier": 2.0, "breakeven_arm_atr_multiplier": 1.0,
    },
}


def _apply(overrides: dict) -> None:
    for name, value in overrides.items():
        current = getattr(settings, name)
        setattr(settings, name, type(current)(value) if not isinstance(current, bool)
                else str(value).lower() in ("1", "true", "yes"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["nordic", "eu", "us"], required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="current")
    parser.add_argument("--set", action="append", default=[], help="override a setting, name=value")
    parser.add_argument("--out", help="write the full result JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("yfinance", "src.analysis.screener", "src.agent.risk", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _apply(PRESETS[args.preset])
    _apply(dict(item.split("=", 1) for item in args.set))

    from src.backtesting.engine import BacktestEngine
    from src.data.watchlist import get_eu_watchlist, get_omxs30_tickers, get_us_tickers
    from src.scheduler.scan_loop import _to_sek_price

    tickers = {"nordic": get_omxs30_tickers, "eu": get_eu_watchlist, "us": get_us_tickers}[args.market]()
    engine = BacktestEngine(
        market=args.market, tickers=tickers,
        start=date.fromisoformat(args.start), end=date.fromisoformat(args.end),
        n_windows=args.windows,
        sek_factor=lambda ticker: _to_sek_price(1.0, ticker, args.market, strict=True),
    )
    result = engine.run().to_dict()

    label = f"{args.market} {args.preset} {' '.join(args.set)}".strip()
    print(f"\n=== {label}: {args.start} → {args.end}, {len(tickers)} tickers")
    for w in result["windows"]:
        m = w["metrics"]
        print(f"  {w['start']} → {w['end']}: {m['total_trades']:4d} trades  win {m['win_rate']:.0%}  "
              f"exp {m['avg_rrr']:+.2f}R  return {m['total_return_pct']:+.1f}%  maxDD {m['max_drawdown_pct']:.1f}%")
    m = result["overall_metrics"]
    print(f"  overall: {m['total_trades']} trades  win {m['win_rate']:.0%}  expectancy {m['avg_rrr']:+.2f}R  "
          f"median hold {m.get('median_hold_days', 0):.0f}d")
    for reason, stats in m.get("exit_reasons", {}).items():
        print(f"    {reason:<16} {stats['n']:5d}  {stats['avg_r']:+.2f}R")
    if args.out:
        Path(args.out).write_text(json.dumps({"label": label, **result}, indent=1))


if __name__ == "__main__":
    main()
