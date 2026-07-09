# Historical replay — prompt optimization on past data

Bootstraps MIPRO training data from history instead of waiting months for live
trades. Fully **separate from the live system**: nothing here is imported by the
scan loop, and compiled programs land in `compiled/backtest/`, which the live
`DecisionEngine` never reads.

## How it works

For each historical scan day (every Nth trading day), the replay:

1. Slices daily OHLCV to everything ≤ that day (point-in-time, no look-ahead)
2. Runs the **same** `technical.py` → `regime.py` → `screener.py` stack as live
3. Reconstructs the news that existed at that day's market close
   (Finnhub for US when keyed, GDELT otherwise — free, both markets, Swedish coverage)
4. Rebuilds a macro context from index/VIX/rates/FX histories (previous close)
5. Simulates what a disciplined BUY would have realized — entry at next open,
   1.5×ATR stop, RRR target, timeout exit, live cost model (slippage + courtage + FX)
6. Writes the five prompt inputs (exactly as `DecisionEngine.decide()` builds
   them) plus the realized `pnl_pct` as one JSONL example

No LLM calls during generation (unless `--news-mode analyzed`). The expensive
calls happen only inside MIPRO, same as the weekly live run.

## Usage

```bash
# 1. Generate ~a year of examples for both markets (idempotent — resumes by key)
python -m src.backtesting generate --start 2025-07-01 --end 2026-06-01

# 2. Inspect class balance / exit reasons
python -m src.backtesting stats

# 3. Optimize a track's prompt offline (needs API keys in .env)
python -m src.backtesting optimize --track claude
python -m src.backtesting optimize --track gpt

# 4. Adopt manually if you like the result
cp compiled/backtest/claude_trade_decision.json compiled/claude_trade_decision.json
```

Useful knobs on `generate`: `--stride` (scan every Nth trading day, default 2),
`--cooldown` (trading days before re-sampling a ticker, default 10),
`--max-hold` (timeout exit, default 20), `--news-mode raw|analyzed|off`
(`analyzed` runs the live `analyze_news` GPT call per candidate for a
distribution-faithful `news_summary`; `raw` formats headlines directly).

All fetched data is disk-cached under `data/backtest/` (gitignored), so reruns
and resumed generations are free.

## Known caveats (accepted, documented, not bugs)

- **Model hindsight leakage** — the decision models' training data likely covers
  the replay period, so MIPRO scores on historical examples are optimistic.
  Headline timestamps are rendered as relative ages ("3d ago") rather than
  dates to weaken the anchor, but the live paper-trading comparison remains the
  real scoreboard. Use this for training-data volume, not performance claims.
- **Survivorship bias** — the watchlists are today's universe constituents.
  Keep the replay horizon to ~6–12 months.
- **News distribution** — `raw` mode feeds a headline list where live feeds an
  LLM summary; use `--news-mode analyzed` when optimizing for real.
- **No earnings filter** — historical earnings calendars aren't freely
  available, so replay candidates aren't earnings-screened like live ones.
- **Macro approximation** — live macro context comes from FRED/Riksbank/ECB;
  replay reconstructs a compact snapshot from VIX/index/10Y/USD-SEK histories.
- **Same-bar ambiguity** — when a daily bar touches both stop and target, the
  stop is assumed to fill first (conservative labeling).
