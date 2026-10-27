"""Post-mortem of closed paper trades and decisions, read straight from the SQLite DB.

Stdlib only, so it runs on the Pi without the venv:
    python3 scripts/trade_postmortem.py                 # data/deepswing.db
    python3 scripts/trade_postmortem.py path/to/backup.db --trades
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "deepswing.db"
ATR_RE = re.compile(r"ATR\(14\):\s*([0-9.]+)")
PRICE_RE = re.compile(r"Price:\s*([0-9.]+)")
RSI_RE = re.compile(r"RSI\(14\):\s*([0-9.]+)")
VOL_RE = re.compile(r"Volume Ratio \(vs 20-avg\):\s*([0-9.]+)x")


def _num(pattern: re.Pattern, text: str) -> float | None:
    m = pattern.search(text or "")
    return float(m.group(1)) if m else None


def _dt(value: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(value) if value else None
    except ValueError:
        return None


def enrich(t: dict) -> dict:
    qty, entry, exit_, stop = t["quantity"], t["entry_price"], t["exit_price"], t["stop_loss"]
    pnl = (exit_ - entry) * qty - (t.get("commission") or 0.0)
    risk = (entry - stop) * qty
    snap = t.get("technical_snapshot") or (t.get("entry_inputs") or {}).get("technicals", "")
    atr, price = _num(ATR_RE, snap), _num(PRICE_RE, snap)
    atr_frac = atr / price if atr and price else None
    stop_frac = (entry - stop) / entry if entry else None
    target_frac = (t["target"] - entry) / entry if entry else None
    entry_t, exit_t = _dt(t.get("entry_time")), _dt(t.get("exit_time"))
    return {
        **t,
        "pnl": pnl,
        "pnl_pct": pnl / (entry * qty) if entry * qty else 0.0,
        "R": pnl / risk if risk > 0 else 0.0,
        "stop_pct": stop_frac,
        "stop_atr": stop_frac / atr_frac if stop_frac and atr_frac else None,
        "planned_rrr": target_frac / stop_frac if stop_frac else None,
        "rsi": _num(RSI_RE, snap),
        "vol_ratio": _num(VOL_RE, snap),
        "atr_pct": atr_frac,
        "hours": (exit_t - entry_t).total_seconds() / 3600 if entry_t and exit_t else None,
        "entry_dt": entry_t,
    }


def _mean(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):+.2f}" if xs else "  n/a"


def _median(xs: list, fmt: str = "{:.2f}") -> str:
    xs = [x for x in xs if x is not None]
    return fmt.format(statistics.median(xs)) if xs else "n/a"


def report_trades(rows: list[tuple], show_trades: bool) -> None:
    for track, closed_json, open_json, cash, start in rows:
        trades = [enrich(t) for t in json.loads(closed_json or "[]")]
        live = [t for t in trades if t.get("exit_reason") != "paused"]
        opened = json.loads(open_json or "[]")
        print(f"\n=== {track}: {len(trades)} closed ({len(trades) - len(live)} paused exits excluded), "
              f"{len(opened)} open, cash {cash:,.0f} / start {start:,.0f}")
        if not live:
            continue
        wins = [t for t in live if t["pnl"] > 0]
        print(f"win rate {len(wins)}/{len(live)} = {len(wins) / len(live):.0%}   "
              f"expectancy {_mean([t['R'] for t in live])}R   "
              f"total net P&L {sum(t['pnl'] for t in live):+,.0f} SEK")
        first, last = min(t["entry_dt"] for t in live if t["entry_dt"]), max(t["entry_dt"] for t in live if t["entry_dt"])
        print(f"entries {first:%Y-%m-%d} → {last:%Y-%m-%d}")

        print("\nby exit reason          n   avg R   med hours held")
        by_reason = defaultdict(list)
        for t in live:
            by_reason[t["exit_reason"] or "?"].append(t)
        for reason, ts in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            print(f"  {reason:<18} {len(ts):>4}  {_mean([t['R'] for t in ts]):>6}   {_median([t['hours'] for t in ts], '{:.1f}'):>6}")

        print("\nby market               n   wins   avg R")
        by_market = defaultdict(list)
        for t in live:
            by_market[t["market"]].append(t)
        for market, ts in sorted(by_market.items()):
            print(f"  {market:<18} {len(ts):>4}  {sum(t['pnl'] > 0 for t in ts):>4}  {_mean([t['R'] for t in ts]):>6}")

        print("\nentry shape (medians):  "
              f"stop {_median([t['stop_pct'] for t in live], '{:.2%}')} of price = "
              f"{_median([t['stop_atr'] for t in live])}×ATR | "
              f"planned RRR {_median([t['planned_rrr'] for t in live])} | "
              f"RSI {_median([t['rsi'] for t in live], '{:.0f}')} | "
              f"vol ratio {_median([t['vol_ratio'] for t in live])}× | "
              f"ATR {_median([t['atr_pct'] for t in live], '{:.2%}')} of price")
        stopped_fast = [t for t in live if t["exit_reason"] == "stop_loss" and t["hours"] is not None and t["hours"] < 24]
        print(f"stop-outs within 24h of entry: {len(stopped_fast)}/{sum(t['exit_reason'] == 'stop_loss' for t in live)}")
        print(f"regimes at entry: {dict(Counter(t.get('regime') or '?' for t in live))}")

        if show_trades:
            print("\n  id  ticker        mkt     entry(UTC)        hours  exit_reason      R      stop%  stopATR  RRR   RSI")
            for t in sorted(live, key=lambda t: t["entry_dt"] or datetime.min):
                print(f"  {t['trade_id']:>3} {t['ticker']:<13} {t['market']:<6} "
                      f"{t['entry_dt']:%m-%d %H:%M}  {t['hours'] or 0:>7.1f}  {t['exit_reason']:<15} "
                      f"{t['R']:>+5.2f}  {(t['stop_pct'] or 0):>6.2%}  {t['stop_atr'] or 0:>6.2f}  "
                      f"{t['planned_rrr'] or 0:>4.1f}  {t['rsi'] or 0:>4.0f}")


def report_decisions(con: sqlite3.Connection) -> None:
    cur = con.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM decisions")
    lo, hi, n = cur.fetchone()
    print(f"\n=== decisions: {n} rows, {lo} → {hi}")
    for track, action, count in con.execute(
        "SELECT track, action, COUNT(*) FROM decisions GROUP BY track, action ORDER BY track, action"
    ):
        print(f"  {track:<8} {action:<8} {count}")
    print("\nregime labels on all decisions:")
    for regime, count in con.execute("SELECT regime, COUNT(*) FROM decisions GROUP BY regime ORDER BY 2 DESC"):
        print(f"  {regime}: {count}")
    print("\ntop block reasons:")
    reasons = Counter()
    for (reason,) in con.execute("SELECT block_reason FROM decisions WHERE action = 'BLOCKED'"):
        reasons[re.sub(r"[0-9.]+", "#", reason or "?")[:90]] += 1
    for reason, count in reasons.most_common(8):
        print(f"  {count:>5}  {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", nargs="?", default=str(DEFAULT_DB))
    parser.add_argument("--trades", action="store_true", help="list every closed trade")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT track, closed_trades, open_positions, cash, starting_equity FROM portfolio_state ORDER BY track"
    ).fetchall()
    report_trades(rows, args.trades)
    report_decisions(con)


if __name__ == "__main__":
    main()
