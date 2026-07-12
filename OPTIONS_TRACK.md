# DeepOptions — Options Trading Track (Design / Feasibility)

Status: **design only — nothing implemented**. This doc answers "can we build a third
simulation vertical that trades options instead of stocks, without a paid options data
feed?" (yes) and sketches the architecture for it.

---

## Feasibility summary

- **Chain data is free.** `yfinance` (already a core dependency) exposes full US option
  chains with no API key: available expiries via `Ticker.options`, and per-expiry
  strike/bid/ask/last/volume/openInterest/impliedVolatility via `Ticker.option_chain(expiry)`.
  The system discovers "what's purchasable" at scan time, the same way it discovers prices.
  No static options knowledge needs to be baked in.
- **US-only.** yfinance has no chains for `.ST` tickers and there is no free source for
  Nasdaq Stockholm derivatives. The options track runs on `us_watchlist` during the
  existing US session (15:30–22:00 CET). Nordic is out of scope (a synthetic
  Black-Scholes market for Nordic is possible but loses real spreads/IV/OI, making
  comparisons dishonest — not worth it).
- **Quote quality caveats.** yfinance chains are ~15-min delayed and illiquid strikes go
  stale. Mitigations: fill at mid-price with adverse half-spread slippage, and hard
  liquidity gates (open interest, volume, max relative spread) before a contract is even
  shown to the model.
- **Greeks aren't provided** (only IV). Delta/theta are computed locally with a
  closed-form Black-Scholes using `math.erf` — pure math, no scipy, Pi-safe.

---

## Where it lives

**Inside DeepSwing as two new tracks** — `claude-opt` and `gpt-opt` — not a third repo.

Rationale: the scan skeleton (fetch → analyze → screen → decide → trade), news pipeline,
macro context, technical/regime analysis, scheduler, persistence, ERL, MIPRO, and
dashboard plumbing all transfer unchanged. A fork would duplicate ~70% of the codebase
and drift. The genuinely new code is confined to:

```
src/data/options_chain.py        chain fetch + liquidity filter + shortlist builder
src/analysis/options_math.py     Black-Scholes price/delta/theta (closed-form, ~40 lines)
src/agent/options_decision.py    DSPy OptionTradeDecision / OptionExitDecision + engine
src/agent/options_risk.py        premium-based sizing + contract validation
src/portfolio/options_simulator.py  OptionsPortfolio / OptionPosition; expiry handling
src/scheduler/options_scan.py    options variant of the scan cycle (US session only)
```

Track plumbing that already generalizes: the DB `track` column, `heuristics/{track}/`,
`compiled/{track}_*.json`, and `portfolio_state` rows are all keyed by track string —
`claude-opt` / `gpt-opt` slot in without schema changes. `settings.tracks` grows an
options counterpart (`options_tracks: list[str] = []`, empty = feature off).

Scope for v1: **long single-leg calls and puts only.** No spreads, no short options, no
assignment mechanics (long-only means exercise/expiry is the only terminal event and
max loss = premium paid). Verticals are a possible phase 2; short premium is out —
undefined-risk positions don't fit the 1%-risk philosophy or a paper simulator this size.

---

## Data layer — `src/data/options_chain.py`

```python
@dataclass
class OptionContract:
    contract_symbol: str      # OCC symbol from yfinance, unique ID
    underlying: str
    right: Literal["call", "put"]
    strike: float
    expiry: date
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    dte: int
    # computed locally:
    mid: float
    spread_pct: float         # (ask - bid) / mid
    delta: float              # Black-Scholes from mid IV
    theta_per_day: float

def fetch_chain_shortlist(ticker: str, spot: float, direction: str) -> list[OptionContract]:
    """Fetch chain, filter for liquidity + swing suitability, return <=8 contracts."""
```

Shortlist rules (all knobs in `config/settings.py`):

| Filter | Default | Why |
|---|---|---|
| DTE window | 21–60 days | Swing horizon + theta buffer; skip weeklies and LEAPS |
| Expiries kept | 2 nearest inside window | Bounds prompt size |
| \|delta\| band | 0.35–0.65 | Near-the-money — directional exposure without lottery tickets |
| Min open interest | 200 | Realistic tradability |
| Min day volume | 10 | Not a dead strike |
| Max spread_pct | 8% | Wide spreads eat the whole swing edge |

`direction` comes from the existing screener/regime signal (bullish setup → calls,
bearish → puts if we ever add bearish entries; v1 mirrors the stock track's long-only
bias, so calls). Result: ~4–8 contracts per candidate, small enough to inline in the
decision prompt.

Cost control on the Pi: chains are fetched **only for screened candidates** (max
`max_candidates_per_session`, 15), never for the whole watchlist — one
`option_chain()` call per candidate per expiry, cached for the scan.

---

## Decision layer — `src/agent/options_decision.py`

New DSPy signature; same engine pattern as `DecisionEngine` (per-track LM via
`build_lm`, compiled program from `compiled/{track}_option_decision.json`, MIPRO-able).

```python
class OptionTradeDecision(dspy.Signature):
    """
    You are evaluating whether to buy a call option on a stock with a bullish swing
    setup. You may only choose from the contracts listed — each line has an index,
    strike, expiry, DTE, mid price, delta, theta/day, IV, OI and spread.

    Return BUY only for a high-conviction setup where the expected underlying move
    exceeds the premium's breakeven within the DTE window. Theta decay is a real
    cost: prefer contracts whose DTE comfortably exceeds the expected swing duration.
    Return PASS if no listed contract offers a sound payoff.
    """
    technicals: str = dspy.InputField(...)
    regime: str = dspy.InputField(...)
    news_summary: str = dspy.InputField(...)
    macro_context: str = dspy.InputField(...)
    heuristics: str = dspy.InputField(...)
    option_shortlist: str = dspy.InputField(desc="Numbered list of purchasable contracts")

    action: Literal["BUY", "PASS"] = dspy.OutputField(...)
    contract_index: int = dspy.OutputField(desc="Index of chosen contract from the list")
    confidence: float = dspy.OutputField(...)
    profit_target_pct: float = dspy.OutputField(desc="Close at +X% of premium, e.g. 0.6")
    max_loss_pct: float = dspy.OutputField(desc="Close at -X% of premium, e.g. 0.4")
    time_stop_dte: int = dspy.OutputField(desc="Close when DTE falls to this, regardless")
    reasoning: str = dspy.OutputField(...)
```

Key differences vs `TradeDecision`:

- The model **picks from a pre-filtered shortlist by index** — it never invents a
  contract, and a bad index is a validation reject. Never feed it a raw 400-row chain.
- Exit levels are **percent-of-premium**, not underlying price levels — option P&L is
  what the simulator marks, and premium-relative stops are how the position is managed.
- **`time_stop_dte` replaces the trailing stop concept.** Options die; a position that
  drifts sideways must be force-closed before gamma/theta destroy it. Risk layer clamps
  it to ≥ 7.

`OptionExitDecision` mirrors `ExitDecision` (HOLD/SELL) with the position context
extended by current premium, P&L% of premium, DTE remaining, and current delta/theta.

Track/heuristic/MIPRO wiring is identical to the stock tracks: `heuristics/claude-opt/`,
30+ closed trades before MIPRO runs, same Sunday 02:00 slot.

---

## Risk layer — `src/agent/options_risk.py`

ATR stops and RRR-from-stop-distance don't map to long options. The translation:

| Stock-track rule | Options-track equivalent |
|---|---|
| 1% equity at risk per trade (stop distance × qty) | Premium paid ≤ 1% of equity (hard cap 2%) — max loss on a long option **is** the premium |
| Stop-loss at 1.5× ATR | `max_loss_pct` of premium (model-chosen, clamped to 30–60%) |
| Min RRR 2.0 | `profit_target_pct / max_loss_pct ≥ 2.0`, **and** a breakeven sanity check: underlying move needed to break even at expiry ≤ 1.5× ATR₁₄ × √(DTE) heuristic budget |
| Drawdown mode halves size | Same — halve contract count (min 1 → reject if 1 is already over budget) |
| No duplicate tickers | No duplicate **underlyings** across open option positions; also reject if the stock tracks' sector-concentration equivalent trips |
| VIX ≥ 35 halts entries | Keep — elevated IV also means overpriced premium |

Sizing: `contracts = floor(risk_budget_sek / (mid × 100 × usd_sek))`. One contract
minimum; if one contract exceeds the hard cap, reject (deep ITM / high-priced
underlyings will legitimately be unaffordable at 100k SEK equity — that's correct
behaviour, not a bug).

Validation also re-checks the chosen contract's liquidity gates at decision time (the
shortlist filter ran pre-prompt; quotes may have moved).

---

## Simulator — `src/portfolio/options_simulator.py`

`OptionsPortfolio` follows the `Portfolio` pattern (same persistence hook, same
`export_state`/`import_state` shape, same `portfolio_state` table) with these deltas:

- **`OptionPosition`**: contract identity (OCC symbol, underlying, right, strike,
  expiry), `contracts` count, ×100 multiplier, `entry_premium`, `current_premium`,
  the three exit knobs, and the entry technical snapshot (same ERL capture need as
  stocks). Market value = `current_premium × 100 × contracts` (converted to SEK the
  same way stock legs are).
- **Fills**: at mid ± half the quoted spread (adverse), replacing the flat
  `simulated_slippage` — option spreads are the dominant friction and we have the real
  spread in hand. Commission per contract (flat SEK amount, Montrose-style) instead of
  percentage courtage.
- **Mark-to-market needs quotes, not just spot.** Each scan refreshes premiums for open
  positions from the chain (one targeted `option_chain()` call per open underlying).
  If a quote is missing/stale, fall back to Black-Scholes off spot + last known IV so
  the equity curve never freezes.
- **`update_premiums()`** replaces `update_prices()`: checks `profit_target_pct`,
  `max_loss_pct`, and `time_stop_dte` (exit reasons `"profit_target"`, `"premium_stop"`,
  `"time_stop"`).
- **Expiry is a hard structural difference from stocks**: a position can terminate with
  no scan running. A daily job after US close (`22:05 CET`) sweeps open positions:
  contracts at expiry close at intrinsic value (`max(spot - strike, 0)` for calls,
  exit reason `"expired_itm"` / `"expired_worthless"`). Long-only means this is pure
  bookkeeping — no assignment, no margin.

`ClosedOptionTrade` carries the same ERL payload (entry inputs, snapshot, reasoning)
plus option-specific fields (IV at entry vs exit, DTE consumed, underlying move vs
premium P&L) — that last pair is exactly what ERL should learn from ("right on
direction, wrong on timing/theta" is *the* canonical options lesson).

---

## Scheduler & dashboard

- `options_scan.py` registers on the same 15-min cadence, gated by
  `is_market_open("us")` — no new market-hours logic.
- Reuses the scan-level plumbing: `_scan_lock` (options scan serializes with stock
  scans since both hit yfinance + LLMs), capacity-aware skip
  (`min_cash_for_new_position_pct`), news cache, macro context.
- Expiry sweep: new daily APScheduler job, 22:05 CET weekdays.
- Dashboard: `/api/comparison` grows the two new tracks; equity curves are
  SEK-denominated so they overlay the stock tracks directly on the existing chart.
  A four-way comparison (Claude-stock vs GPT-stock vs Claude-opt vs GPT-opt) on
  identical market data is the payoff of doing this in-repo.

---

## Build order

1. `options_math.py` + `options_chain.py` with a standalone smoke script (fetch AAPL
   shortlist, print it) — proves the data layer before anything depends on it.
2. `options_simulator.py` + expiry sweep, unit-tested against hand-computed fills and
   expiry cases (this is the module where silent bugs cost the most).
3. `options_risk.py` (pure function, easy tests).
4. `options_decision.py` + prompt shortlist formatting.
5. `options_scan.py` wiring + settings + dashboard track registration.
6. Let it run ≥30 closed trades → first MIPRO compile for the options signatures.

Open questions (decide before step 5):

- Starting capital for options tracks: same 100k SEK, or smaller (options compound
  faster in both directions)?
- Should puts be enabled in v1 (requires a bearish screener path that doesn't exist for
  stocks), or calls-only to mirror the long-only stock tracks? Recommendation: calls-only
  first; a bearish path changes the screener, not just the options code.
