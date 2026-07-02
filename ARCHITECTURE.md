# DeepSwing — Architecture & Design

## System Flow (every 15 min during market hours)

```
SCHEDULER → is_market_open(nordic | us)?
  ↓
DATA FETCHER (shared by both tracks)
  ├── OHLCV: yfinance for US; Alpha Vantage (.STO) for Nordic; yfinance (.ST) as fallback
  ├── News: NewsAPI + Swedish RSS (DI.se, Börsdata)
  ├── Insider: SEC EDGAR (US) / FI Insynsregistret (SE)
  └── Macro: FRED (US) / Riksbank + ECB (Nordic)
  ↓
TECHNICAL ANALYSIS — shared
  ├── Trend: 21 EMA, 50 SMA, 200 SMA
  ├── Volatility: 14-period ATR, Bollinger Bands (20, ±2σ)
  ├── Momentum: RSI (14), Parabolic SAR, Ease of Movement
  ├── Volume: OBV, volume ratio vs 20-period avg
  └── Structure: Fibonacci 38.2% / 61.8% retracement (50-bar swing)
  ↓
REGIME CLASSIFIER — shared
  ├── Hurst Exponent (rolling 100-bar R/S analysis)
  │   H > 0.55 → trending → EMA crossover / breakout tactics
  │   H < 0.45 → mean-reverting → Bollinger Band bounce tactics
  └── Lag-1 autocorrelation of log returns
  ↓
SCREENER → 5-10 candidates — shared output
  ├── Price above 50 SMA (bullish structure)
  ├── RSI 40–65 (not extended on entry)
  ├── Volume ≥ 1.2× 20-day avg (measured on the last *completed* daily bar)
  └── Regime-appropriate setup (skip neutral regime)
  ↓
FOR EACH CANDIDATE × EACH TRACK ("claude", "gpt"):
  1. HeuristicStore retrieves top-5 track-specific rules by relevance score
  2. claude-sonnet-5 / gpt-5 decide via DSPy TradeDecision (shared gpt-5-mini news analysis)
  3. Risk Engine validates: ATR stop (currency-safe), RRR ≥ 2.0, 1% risk size,
     position value ≤ 25% equity & ≤ cash, no duplicate ticker, sector cap,
     pairwise return correlation ≤ 0.7 vs open positions
  4. Portfolio Simulator records fill with 0.05% simulated slippage
  ↓
POSITION MONITOR (every 15 min, per track)
  ├── Stop-loss hit → close + trigger ERL
  ├── Take-profit hit → close + trigger ERL
  ├── Trailing stop update (2×ATR trail once in profit → exit_reason="trailing_stop")
  └── News-exit review when a holding moves ≥5% since its last news check
  ↓
DASHBOARD WebSocket push (both tracks)
```

---

## Dual Simulation Tracks

```
SHARED DATA                    SHARED SCREENER
     ↓                               ↓
┌────────────────┐         ┌────────────────┐
│  CLAUDE TRACK  │         │   GPT TRACK    │
│                │         │                │
│ claude-sonnet-5│         │ gpt-5          │
│ (decisions)    │         │ (decisions)    │
│                │         │                │
│ claude-opus-4-8│         │ gpt-5.5        │
│ + ext.thinking │         │ (ERL)          │
│ (ERL)          │         │                │
│                │         │                │
│ compiled/      │         │ compiled/      │
│ claude_*.json  │         │ gpt_*.json     │
│                │         │                │
│ heuristics/    │         │ heuristics/    │
│ claude/        │         │ gpt/           │
└────────────────┘         └────────────────┘
       ↓                          ↓
  Portfolio A               Portfolio B
       ↓                          ↓
           DASHBOARD: head-to-head comparison
```

Both tracks start with 100,000 SEK simulated capital. All records in the DB are tagged with a `track` column.

---

## DSPy TradeDecision Signature

```python
class TradeDecision(dspy.Signature):
    """Evaluate a potential LONG ENTRY — BUY only on a high-conviction setup."""
    technicals: str     = dspy.InputField()   # all TA indicators
    regime: str         = dspy.InputField()   # Hurst + tactic recommendation
    news_summary: str   = dspy.InputField()   # shared gpt-5-mini per-ticker analysis
    macro_context: str  = dspy.InputField()   # FRED / Riksbank rates
    heuristics: str     = dspy.InputField()   # top-5 ERL rules for this context

    action: Literal["BUY", "PASS"] = dspy.OutputField()
    confidence: float   = dspy.OutputField()   # 0.0 – 1.0
    stop_loss: float    = dspy.OutputField()   # must be below entry for BUY
    target: float       = dspy.OutputField()   # must give RRR ≥ 2.0
    reasoning: str      = dspy.OutputField()   # references specific signals
```

Starts uncompiled on day one. After ~30 closed trades per track, weekly MIPROv2 optimization compiles improved prompt instructions + few-shot examples, saving `compiled/{track}_trade_decision.json`. Each track compiled independently.

---

## ERL (Experiential Reflective Learning)

```
TRADE CLOSED
  ↓
EVALUATOR: logs entry/exit, P&L%, regime, signals, stop-hit flag
  ↓
CAUSAL ANALYSIS MODEL:
  Claude track → claude-sonnet-4-6 with extended_thinking=True
  GPT track    → gpt-4o (standard)
  
  Extracts trigger-action heuristic:
    "IF [specific condition] THEN [what to do/avoid]"
    + quality score 0–10 + market + regime tags
  ↓
HEURISTIC saved to heuristics/{track}/*.json
  ↓
WEEKLY MAINTENANCE:
  - Prune: quality < 4 AND access_count < 2
  - Promote: access_count ≥ 10 → mark as core rule
  - MIPRO optimization: recompile DSPy program for each track
```

Heuristic example:
```json
{
  "id": "uuid",
  "trigger": "breakout candle on daily, volume < 1.5× avg",
  "action": "DO NOT enter — high fake-out probability when volume doesn't confirm",
  "market": "us",
  "regime": "trending",
  "quality_score": 8.2,
  "access_count": 7,
  "is_core": false,
  "created": "2026-06-26"
}
```

---

## Risk Rules

| Parameter | Value |
|---|---|
| Starting capital (per track) | 100,000 SEK |
| Max risk per trade | 1% of portfolio |
| Hard cap per trade | 2% of portfolio |
| Minimum RRR | 2.0 |
| Stop-loss | 1.5 × ATR below entry (validated as fraction of price) |
| Position value cap | 25% of equity, and ≤ available cash |
| Trailing stop | 2 × ATR once in profit (closes as `trailing_stop`) |
| Slippage (simulated) | 0.05% |
| Drawdown pause | >10% → halve position size + audit |
| Max sector correlation | 0.7 |

---

## Market Hours

| Session | Open (CET) | Close (CET) | Scan window |
|---|---|---|---|
| Nordic (Stockholm) | 09:00 | 17:30 | 08:45 – 17:45 |
| US (NYSE/NASDAQ) | 15:30* | 22:00* | 09:15 – 16:15 ET |
| Overlap | 15:30 | 17:30 | both active |

*US hours are evaluated in **US Eastern Time** (9:30–16:00 ET); the CET equivalents shift during the ~3 weeks/year when US and EU DST are out of sync.

Scheduler: APScheduler, 15-min interval, `Europe/Stockholm` timezone; nightly SQLite snapshot 23:45, weekly MIPRO + housekeeping Sunday 02:00.

---

## Data Sources

| Type | Source | Notes |
|---|---|---|
| Nordic OHLCV | Alpha Vantage (`.STO`) | 25 req/day free — batched at open |
| Nordic OHLCV fallback | yfinance (`.ST`) | Covers most OMXS30 |
| US OHLCV | yfinance | Batch download, free |
| News (EN) | NewsAPI | 100 req/day free |
| News (SE) | DI.se RSS, Börsdata RSS | Free, no auth |
| Insider (US) | SEC EDGAR API | Free, public |
| Insider (SE) | FI Insynsregistret CSV | Free daily export |
| Macro (US) | FRED API | Free, needs key |
| Macro (SE/EU) | Riksbank API, ECB data API | Free, public |

---

## Directory Structure

```
DeepSwing/
├── src/
│   ├── data/           market_data, news_fetcher, insider_fetcher, macro_data
│   ├── analysis/       technical (11 indicators), regime (Hurst), screener
│   ├── agent/          decision (DSPy), risk, memory (ERL store), erl, news_analyzer
│   ├── portfolio/      simulator (paper trades), metrics (Sharpe, drawdown, win rate)
│   ├── scheduler/      market_hours, scan_loop (main cycle), optimizer (MIPRO)
│   └── dashboard/      FastAPI app, Chart.js frontend
│       ├── static/     app.js, style.css
│       └── templates/  index.html
├── config/             settings.py (Pydantic Settings, reads .env)
├── data/               deepswing.db (SQLite — gitignored)
├── heuristics/         claude/ and gpt/ subdirs (JSON rules — gitignored)
├── compiled/           MIPRO-compiled DSPy programs (gitignored)
├── systemd/            deepswing.service
├── main.py             entry point (scheduler + uvicorn)
├── requirements.txt
├── .env.example
├── SETUP.md
├── ARCHITECTURE.md     (this file)
└── STATUS.md
```
