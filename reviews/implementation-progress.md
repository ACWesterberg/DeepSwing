# Learning-system improvements: implementation progress

## Completed: correctness pass, 2026-09-21

The first implementation addresses measurement and execution defects from the [review](deep-dive-2026-09-21.md). It does not establish that any trading prompt is profitable or improved.

| Review area | Implementation |
|---|---|
| Replay failures | Explicit track model, no silent PASS for errors/invalid actions, declared input fields only, old corpus versions rejected. |
| Counterfactual labels | Shared daily executor with backtests, commissions/FX fees/slippage, gaps, trailing stops, breakeven, net-R labels, small outcomes retained. |
| Backtest chronology | Existing holdings advance before new close-price entries; new positions cannot use earlier intraday extremes. |
| Accounting | Backtest returns/R/win rate use net proceeds; horizon exits count; expectancy matches live definition; open equity changes appear before the first close. |
| Risk | Finite positive input checks, drawdown reduction after sizing caps, both fee legs and slippage in the cost floor, invalid live quotes cannot corrupt portfolio state. |
| Learned rules | Negations/comparison directions are preserved; measured outcomes are required for new core promotions; real and hypothetical feedback use the same percentage-return unit. |
| Exit order | Mechanical stops/targets execute before optional LLM reviews; stale entry technicals are labeled explicitly. |
| Prompt persistence | Validate a staged JSON program before replacing the incumbent; preserve the previous file on save/load failures and archive successful replacements. |
| Sample order | Sort sampled counterfactuals by decision time before creating examples. |

The extracted `DailyPortfolio` is the common implementation for daily-bar execution. The backtest preserves its idealized close-price entry convention; it does not claim next-open execution or equivalence to 15-minute live polling. Existing holdings process the day's bar before new entries.

Full OHLC data supplies gap information. Historical rows with only a close use a close-only approximation; high/low rows without an open use the previous close as the opening estimate. This fallback is not evidence of intraday execution fidelity. Missing/invalid numerical data cannot generate a successful replay score.

## Replay migration

Old caches used gross or inconsistent outcomes, so the loader requires rebuilding them. Existing portfolios, heuristics and deployed prompts are not rewritten. Historical heuristic scores are not retrospectively recalculated by this change.

```bash
python scripts/replay_decisions.py build --out data/replay_corpus.json
python scripts/replay_decisions.py score --reference
python scripts/replay_decisions.py score --track claude --program baseline
python scripts/replay_decisions.py score --track claude --program compiled/claude_trade_decision.json
```

Building fetches market data; scoring real programs invokes the configured provider. The implementation work did neither. `--track` selects the evaluation model for every program, including saved programs. `COUNTERFACTUAL_BUY_THRESHOLD` no longer controls labels: positive net R is BUY, zero/negative net R is PASS.

## Verification

The unchanged baseline passed 497 tests. After all four implementation passes, **616 tests passed** with:

```bash
.venv/bin/python -m pytest -q --disable-warnings
```

The suite includes regression cases for provider errors, model metadata filtering, failed artifact writes/validation, capped drawdown sizing, non-finite inputs, gap losses, opening target fills before later lows, breakeven, trailing exits, FX costs, cost-only losers, contradictory rules, feedback-unit parity, sampled time order and same-day entry/exit ordering.

`tests_sdk/test_replay_integration.py` runs in a fresh subprocess to avoid the unit suite's SDK mocks. It uses installed DSPy with its local `DummyLM`, verifies baseline and saved-program inference, and blocks socket connections. It skips explicitly if DSPy is absent. The verified environment used Python 3.14 and DSPy 3.3.1. No provider compatibility or Raspberry Pi deployment test is implied.

## Completed: temporal evaluation and prompt promotion

Every training example now records its decision time, label-availability time, source and identifier. Real and counterfactual examples share global 60/20/20 time boundaries. Training and validation rows whose outcomes overlap the following period are purged; test labels must be mature. MIPRO sees only training and validation rows.

Before compiling, the default gate requires 20 training examples (including 10 real trades), 10 validation examples, and 20 fresh test examples spanning 10 tickers and 5 days. Purging can leave fewer examples than the nominal split. Insufficient evidence skips the run before model calls.

The serialized candidate is compared with the incumbent, baseline, always-BUY and always-PASS on identical held-out opportunities. Promotion requires at least 5 BUY predictions, positive total stored R, and a mean metric gain of at least 0.01 against every reference. Paired ticker-cluster and day-cluster bootstrap lower bounds must both exceed zero. This is a conservative screening heuristic, not proof of generalization: it does not fully model cross-ticker/serial dependence or correct for an unlimited sequence of future experiments.

The test-time watermark is reserved before scoring. Rejected candidates and scoring failures consume that period; compilation failures do not. Later runs require newer test decisions. Previous test periods may become historical training/validation data, but cannot be reused as promotion tests. Keep the state file when migrating or restoring artifacts; deleting it discards this protection. The scheduler should run one optimization per track at a time.

Artifacts are stored under `compiled/evaluations/<track>/<run_id>/`: frozen example corpora and hash, incumbent and candidate snapshots, model and label settings, predictions, scores, comparisons and status in `report.json`. `compiled/evaluations/<track>/state.json` retains the watermark. Rejected/error runs preserve the active prompt; successful replacements archive it. Runtime reload and backup failures are recorded separately from an on-disk promotion.

Tests cover overlap purging, shared timestamps, immature labels, reused test periods, minimum diversity, missing references, incomplete scores, rejected/error candidates, corrupt state, compile failures and promotion/reload behavior. The real-DSPy smoke test also exercises metadata and evaluation with actual `dspy.Example` objects.

## Completed: predicted trade-plan evaluation

The scheduled optimizer now builds examples from persisted BUY/PASS/BLOCKED decisions with native entry quote, ATR and all five model inputs. Each example freezes its subsequent OHLC bars, execution settings and execution-code hash. Historical BUYs and skipped opportunities use the same configured horizon. Raw bars and predicted plans are included in promotion evidence; future paths never enter model inputs or instruction-proposal examples.

MIPRO's metric and the promotion gate now simulate predicted stops and targets through the shared daily executor. The risk checks match the live static stop-distance and minimum-RRR constraints. Invalid structured predictions abort held-out scoring; static risk rejections are reported as blocked BUYs with no execution or profit. Search-time malformed outputs receive zero credit. Portfolio cash, concentration, correlation, and discretionary LLM exits are outside this isolated-plan experiment.

Every plan uses the same pre-decision ATR risk unit: net one-share profit divided by `ATR_STOP_MULTIPLIER × ATR`. This prevents a tighter proposed stop from increasing R solely by shrinking its denominator. The bounded score is `0.5 + atan(net_R) / pi`; PASS and blocked trades score 0.5 and earn zero R. This differs intentionally from the older action-label score that credited avoided losses. Reported total R adds isolated opportunities; it is not an achievable portfolio return.

The always-BUY reference uses the frozen fixed ATR plan, while always-PASS takes no trades. A fixed-plan hindsight label is not an upper bound for other stops/targets, so plan mode does not report an oracle. The incumbent and candidate both use their actual predicted plans, with the existing fresh-test promotion requirements.

New successful BUY records preserve quote/ATR/inputs, even when an earlier PASS was stored on the same day. Older BUY records lacking this structure are excluded. This can delay optimization until enough mature real-entry examples accumulate. The previous action-only evaluator remains available for historical diagnosis; scheduled optimization now requires plan evidence.

```bash
python scripts/replay_decisions.py build --track claude --plans --out data/plan_corpus.json
python scripts/replay_decisions.py score --plans --corpus data/plan_corpus.json --reference --out data/plan_references.json
python scripts/replay_decisions.py score --plans --corpus data/plan_corpus.json --track claude --program baseline --out data/plan_baseline.json
```

Building fetches data; model scoring invokes the configured provider. Neither was run against external services during this implementation. Existing version-2 action corpora still load for action scoring, but plan scoring requires rebuilding with `--plans`. Execution-code mismatches fail explicitly rather than silently changing an archived experiment.

Plan mode requires valid full OHLC bars, unique dates strictly after the decision day, and a completed horizon. It rejects histories missing more than seven calendar days at either boundary; this coarse coverage check is not an exchange-calendar completeness guarantee. Entry uses the recorded quote and future daily bars, omitting the remainder of the entry session. Adjusted historical data, corporate actions, and missing intraday paths remain important limits.

Tests cover target and stop sensitivity, a constant risk denominator, frozen fees/settings, gaps, ambiguous bars, live static-risk parity, malformed paths/outputs, model-input isolation, real-DSPy execution, corpus persistence, BUY recording, and plan-based optimizer promotion.

## Completed: portfolio research replay

`score --portfolio --track <track>` reuses frozen predictions and price paths in one chronological cash ledger. It models risk-based sizing, position and market allocation caps, entry costs, duplicate holdings, sector limits where metadata exists, same-market return correlation, and drawdown size reduction after caps. Each position uses the shared daily executor with its frozen fees, slippage, trailing and breakeven settings.

New `build --plans` corpora include a frozen portfolio policy and up to 61 prior daily closes per opportunity. New BUY/PASS/risk-BLOCKED records also preserve `replay_sector` outside model input fields. Missing correlation history blocks an otherwise eligible concurrent same-market entry and is counted explicitly; missing sector metadata follows the live unknown-sector convention and is separately reported. Mixed tracks, conflicting overlapping price histories, missing frozen policy and malformed data fail explicitly.

Entries are processed by recorded time, market and ticker, using prior-session portfolio marks. All same-day entries precede the day's full OHLC exit checks, so unknown intraday exit timing cannot release cash early. Entry-session intraday movement is unavailable. Remaining positions close at their stored horizon. The report contains every entry/rejection, closed-trade accounting, fees and dated NAV, including stale marks and unrealized drawdown. Cash must reconcile with initial capital plus net closed-trade P&L.

Native prices are normalized to a 100-unit entry value per position, assuming constant FX and fractional normalized units. This preserves local-price percentage returns without inventing historical FX rates. Reported equity is a research accounting unit, not reconstructed SEK performance. The frozen default starting equity is 100,000 units; no existing live holdings are imported.

```bash
python scripts/replay_decisions.py build --plans --track claude --out data/plan_corpus.json
python scripts/replay_decisions.py score --portfolio --track claude --corpus data/plan_corpus.json --reference --out data/portfolio_references.json
python scripts/replay_decisions.py score --portfolio --track claude --corpus data/plan_corpus.json --program baseline --program compiled/claude_trade_decision.json --out data/portfolio_comparison.json
```

Model scoring still costs provider calls; references use no LLM. The report freezes portfolio policy, corpus hash, execution-code hash, and predictions. Existing paths need rebuilding following the shared static-validation refactor and to capture portfolio policy/history.

This is an offline diagnostic, not an additional automatic promotion gate. Scheduled promotion remains based on held-out isolated plans. The corpus is a sampled set of persisted opportunities, and reconstructed contexts retain historical heuristic text. Missing FX, sector history, exact market calendars, entry-session bars and discretionary exits limit realism. There is no annualized Sharpe estimate from this sparse event calendar.

Tests verify ledger reconciliation, transaction costs, sizing, market/sector/correlation caps, missing-history behavior, no same-day reuse of exit proceeds, later cash reuse, unrealized drawdown, post-cap drawdown sizing, deterministic ordering, frozen policy, native-price normalization, stale marks, model-input isolation, CLI reporting and real-DSPy portfolio scoring.

## Remaining work

1. Run fixed-prompt comparisons on an archived corpus with documented coverage, then determine whether portfolio replay is reliable enough for a promotion veto. No measured strategy improvement is claimed by the implementation tests.
2. Improve market-data provenance, historical FX and corporate-action handling; expand sampling coverage and compare evolving memory separately from fixed saved heuristic text.
3. Improve regime estimation and structural prompt inputs, and evaluate rules causally before producing the final human-readable instruction/evidence package.

No training, paid inference, market-data fetch, or deployment was performed during implementation. The historical review and reproduction harness describe the original checkout; use regression tests for the corrected implementation.
