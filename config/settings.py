from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""
    finnhub_api_key: str = ""   # optional — preferred US per-ticker news when set
    fred_api_key: str = ""

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    log_level: str = "INFO"

    # Simulation tracks
    tracks: list[Literal["claude", "gpt"]] = ["claude", "gpt"]
    starting_capital_sek: float = 100_000.0

    # Claude models
    claude_decision_model: str = "claude-sonnet-5"          # scan decisions (up from Haiku)
    claude_erl_model: str = "claude-opus-4-8"               # heavy post-trade reasoning
    claude_erl_extended_thinking: bool = True               # adaptive thinking on Opus 4.8
    claude_erl_effort: str = "high"                         # output_config.effort; low|medium|high|max
    claude_prompt_model: str = "claude-opus-4-8"            # MIPRO instruction proposer

    # GPT models
    gpt_decision_model: str = "gpt-5"                        # scan decisions (up from 4o-mini)
    gpt_news_model: str = "gpt-5-mini"                       # shared news analysis (light task)
    gpt_erl_model: str = "gpt-5.6-sol"                       # heavy post-trade reasoning
    gpt_erl_reasoning_effort: str = "high"                   # GPT "thinking" for ERL; "" disables
    gpt_prompt_model: str = "gpt-5.6-sol"                    # MIPRO instruction proposer

    # Reasoning effort on the GPT-5 family. Reasoning tokens are billed as output
    # tokens, and the API default is "medium" — so leaving this unset was the
    # single largest line on the OpenAI bill: every scan decision quietly paid
    # for a thousand-plus invisible tokens on top of its answer.
    #
    # Two tiers, because the tasks are not comparable:
    #   decision — the scan decision (and the MIPRO task model that replays it).
    #     "low" is deliberate, not a downgrade by default: the model is handed a
    #     pre-computed indicator digest, an explicit placement rule and a typed
    #     output, and every BUY is independently re-validated by risk.py. If a
    #     compiled program regresses it shows up grouped by program_hash.
    #   light — triage / news analysis / watch classification. These return a
    #     ticker list, three sentences and a one-word verdict; medium reasoning
    #     on them is pure waste, and on the news model it was spending the
    #     answer budget on thinking (the summary-quality item in STATUS.md).
    #
    # "" sends no parameter at all (provider default) — use it for a
    # non-reasoning model, which rejects the field outright.
    gpt_decision_reasoning_effort: str = "low"    # minimal|low|medium|high; "" = provider default
    gpt_light_reasoning_effort: str = "low"       # triage / news / watch classifier

    # Risk parameters
    max_risk_per_trade: float = 0.01       # 1% of portfolio
    hard_cap_risk_per_trade: float = 0.02  # 2% hard cap
    min_rrr: float = 2.5
    atr_stop_multiplier: float = 1.5
    # The ATR gate used to bound the model's stop from one side only, on the
    # assumption that "too tight a stop is fine". It isn't. A stop inside
    # ordinary intraday noise is a coin flip on the next tick — one live trade
    # was handed a 0.26% stop and was gone in 72 minutes — and because trading
    # costs are fixed, a stop that small is dominated by commission: the round
    # trip on an LSE name is 0.5%, so that stop-out booked -2.77R instead of
    # -1R. Two independent floors, whichever binds harder:
    #   noise — the stop must sit outside normal daily movement
    #   cost  — a stop-out must be dominated by the move, not by commission
    # Neither costs anything in position size: sizing is
    # min(risk/stop_frac, max_position_pct) and the value cap already binds for
    # any stop tighter than max_position_pct.
    min_stop_atr_multiplier: float = 0.5   # >= this many ATRs from entry
    min_stop_cost_multiple: float = 3.0    # >= this many round trips
    # Risk-based sizing alone is unbounded (tight stop → huge position), so position
    # value is also capped as a fraction of equity — and at available cash.
    #
    # This doubles as the throughput knob, and it is the binding one. Concurrent
    # positions ≈ market_allocation ÷ position size, so at 0.25 the book held
    # *two* positions total and a 25% position could not fit the 0.20 EU budget
    # at all — EU was structurally untradeable and nothing logged it. At ~9-day
    # holds that was 135 days to accumulate the 30 trades MIPRO needs. 0.10
    # gives 10 slots (nordic 4 / eu 2 / us 4) and roughly a month.
    #
    # The cost is real but lands somewhere that doesn't matter here: with the
    # cap binding, actual risk per trade falls to ~0.3–0.9% of equity and
    # max_risk_per_trade rarely applies. The learning signal is untouched —
    # the MIPRO metric and rrr_achieved are denominated in R, which is
    # invariant to position size — but the equity curve is a weaker proxy for
    # a book sized at a full 1% risk.
    max_position_pct: float = 0.10
    # Trailing stop distance in ATRs once a position is in profit. Wider than the
    # entry stop (1.5×ATR) so ordinary daily noise doesn't knock out winners
    # before the min_rrr target is reachable.
    trailing_stop_atr_multiplier: float = 2.0
    # Because the trail is wider than the entry stop, `peak - 2×ATR` only clears
    # breakeven after price has run a full 2×ATR — everything below that exits at
    # a loss. An independent breakeven floor arms at this many ATRs of profit and
    # never falls back, so a trade that ran and reversed gives back the move, not
    # the risk. Set to 0 to disable the floor entirely.
    breakeven_arm_atr_multiplier: float = 1.0
    drawdown_pause_threshold: float = 0.10
    max_sector_correlation: float = 0.7
    max_positions_per_sector: int = 2
    vix_halt_threshold: float = 35.0   # halt new entries when VIX >= this
    simulated_slippage: float = 0.0005     # 0.05% bid/ask spread approximation
    # Montrose Premium: 0.10% courtage each way; 0.10% FX fee for non-SEK trades
    commission_pct: float = 0.001          # 0.10% per trade leg (buy + sell)
    fx_commission_pct: float = 0.001       # 0.10% extra on USD/EUR legs (US market)
    # Entries fill at a live quote, not the OHLCV close the candidate was screened
    # on — EOD/delayed feeds make that close hours stale, and booking it while
    # exits fill live realizes the gap as phantom P&L. If the live quote has
    # drifted more than this fraction from the scan price, the screened setup no
    # longer describes the market and the entry is blocked.
    max_entry_price_deviation: float = 0.03

    # Dashboard security
    reset_pin: str = "3821"
    dashboard_user: str = "deepswing"
    dashboard_password: str = ""   # leave empty to disable auth

    # Screener thresholds — loosened to widen the funnel (more at-bats for MIPRO
    # to learn from). The AI decision + min_rrr risk validation remain the quality
    # gate downstream, so this raises trade volume without lowering standards.
    # Hurst estimation basis. R/S on price *levels* (legacy default) biases H
    # upward — a plain drifting random walk reads "trending". On *returns* H
    # measures persistence properly, but drifting walks then read ~0.5 (neutral)
    # and the screener gets much stricter. Flip deliberately and observe.
    hurst_on_returns: bool = False

    rsi_min: float = 35.0                  # was 40.0
    rsi_max: float = 78.0                  # was 70.0 — 70 rejected names mid-move
    volume_spike_multiplier: float = 1.2   # was 1.5 (20% above avg vol, not 50%)
    # Every trade level is denominated in ATR, so a low-ATR name can only pay a
    # small win and its tight stop makes the value cap bind before full risk is
    # deployed. Screen them out rather than sizing around them.
    min_atr_pct: float = 0.02              # reject ATR below 2% of price; 0 disables
    max_candidates_per_session: int = 15   # was 10
    earnings_buffer_days: int = 2          # exclude candidates within N days of earnings
    market_news_max_headlines: int = 20    # market-wide headlines injected into macro context

    # Pre-decision triage: every screened candidate costs a news fetch + news
    # analysis + one decision call per funded track, which dominates LLM spend.
    # One cheap shared call ranks the candidates on technicals and only the top
    # K proceed. Fails open to the screener's own top-K; 0 disables entirely.
    triage_enabled: bool = True
    triage_model: str = "gpt-5-mini"
    triage_keep_top: int = 5

    # Re-decision gate. A candidate that survives the screener keeps surviving it
    # every 15 minutes, so the same ticker was sent to the decision model ~36
    # times a session for an answer driven by *daily* bars. DSPy's own cache
    # never catches this: the live price makes every prompt unique by a few
    # decimals. So gate on materiality instead — a PASS is reused until it goes
    # stale, the price moves, or the news changes underneath it.
    #
    # Only PASS is ever reused. A cached BUY would open a position against a
    # stop and target placed at an older price; re-asking costs one call and is
    # the rare case anyway.
    decision_cache_minutes: int = 60        # 0 disables reuse entirely
    decision_recheck_move_pct: float = 0.015  # re-ask once price moves this far

    # yfinance batch downloads fail above ~200 symbols — chunk large universe
    # watchlists so cold-cache scans still populate every ticker.
    ohlcv_batch_chunk_size: int = 150

    # Scheduler intervals (minutes)
    # Every LLM cost scales linearly with this, and 15 minutes was far denser
    # than the strategy needs: decisions are computed off *daily* bars and the
    # book holds for ~9 days, so 30 buys the same information for half the calls.
    #
    # It is not free, and the cost is not in the decisions. This timer also
    # drives the stop/target sweep, and `update_prices` books an exit at the
    # price it *observes*, not at the level that was breached — so the sampling
    # gap is the exit latency. A stop-out now fills up to 30 minutes past the
    # stop instead of 15, worth roughly 0.05-0.15R of extra loss per stop-out on
    # a 3%-of-price stop. Note that this is an artifact of polling, not of the
    # strategy: a real broker's stop order fills at the level regardless.
    scan_interval_minutes: int = 30
    news_refresh_interval_minutes: int = 60  # also the per-ticker news cache TTL

    # NewsAPI rate-limit resilience: if a per-ticker fetch stalls longer than
    # this (retry/backoff = throttled), trip a breaker that skips NewsAPI for
    # newsapi_cooldown_minutes so the rest of the scan uses RSS only and doesn't
    # stall ~1 min per ticker. Set the threshold to 0 to disable the breaker.
    newsapi_slow_threshold_seconds: float = 8.0
    newsapi_cooldown_minutes: int = 20

    # Fully-allocated behaviour: once a track's free cash falls below this fraction
    # of its equity it can't meaningfully open a new position, so the scan skips the
    # candidate/news/decision pipeline for it. When no track is funded the whole scan
    # drops to a lightweight holdings-only monitor.
    min_cash_for_new_position_pct: float = 0.05
    # Per-market cap on invested value as a fraction of each track's equity, so one
    # market can't consume all the cash and starve the other. The US session is long
    # and scans while Stockholm is closed, so without a cap it fills the book before
    # the Nordic session opens. Each market's open-position value is held below its
    # cap; a market with no entry here (or > 1.0) is uncapped. Keep the values below
    # 1.0 to reserve room — the leftover is what the other market can deploy.
    market_allocation: dict[str, float] = Field(default={"nordic": 0.4, "eu": 0.2, "us": 0.4})
    # Holdings are monitored on price alone; a news pull + AI exit review only fires
    # for a position once it has moved at least this fraction (up or down) since its
    # last news check — a "large jump". Set to 0.0 to review every scan.
    holdings_news_jump_pct: float = 0.05

    # Counterfactual MIPRO training: PASS decisions (persisted with their DSPy
    # inputs + decision-time price) are labeled from what the price actually did
    # over the horizon, so the optimizer also learns from setups it declined —
    # without this the trainset only contains taken trades (survivorship bias).
    counterfactual_horizon_days: int = 14        # calendar days of forward price data
    counterfactual_buy_threshold: float = 0.03   # fwd return >= 3% labels the PASS as a missed BUY
    counterfactual_max_examples: int = 200       # absolute ceiling on counterfactual volume
    # Counterfactuals used to be capped at parity with real trades, which tied
    # the trainset to the scarcest input: a live run discarded 60 of 90 labelled
    # PASS decisions and left MIPRO selecting instructions on a 12-example
    # validation split. PASS decisions accumulate far faster than closed trades
    # and are labelled from price data alone, so cap them as a multiple instead.
    # MIN_REAL_EXAMPLES already stops a trainset that is purely hindsight.
    counterfactual_ratio_cap: float = 4.0        # max counterfactuals per real trade

    # Housekeeping: decisions accumulate ~1k rows/day at 15-min scans; prune rows
    # older than this during weekly maintenance (0 disables). Counterfactual
    # training only reads recent PASSes, so 90 days is generous.
    decisions_retention_days: int = 90
    # Daily on-disk SQLite snapshot (data/backups/), keep the newest N (0 disables).
    # MIPRO programs are backed up offsite; this protects the portfolio DB itself.
    db_backup_keep: int = 7

    # MIPRO artifact backup — path to a local git working copy of a standalone
    # backups repo (e.g. ~/Github/deepswing-mipro-backups). Set via env
    # MIPRO_BACKUP_REPO_DIR. Empty disables backup. The Pi must have push
    # credentials configured on that working copy's remote.
    mipro_backup_repo_dir: str = ""
    mipro_backup_push: bool = True  # commit locally always; push to remote if True

    # Boot-time preflight: ping each configured model once so a bad ID/credential
    # surfaces immediately in the logs instead of at the next scan/ERL/MIPRO run.
    preflight_check_models: bool = True

    # Personal watchlist alerts — dashboard Watchlist tab pings Telegram on large
    # day moves, fresh directional news, and insider activity for tickers the user
    # watches. Alerts no-op (logged only) until both Telegram keys are set.
    telegram_bot_token: str = ""            # from @BotFather
    telegram_chat_id: str = ""              # target chat/user id
    watch_interval_minutes: int = 15
    watch_move_alert_pct: float = 0.03      # day move vs previous close that pings
    # Re-ping only when the move extends this much beyond the last alerted level,
    # so a runaway mover pings again but a stalled one stays quiet.
    watch_move_realert_step_pct: float = 0.02
    watch_insider_buys_only: bool = False   # True = skip bearish insider alerts (sells)
    watch_classifier_model: str = "gpt-5-mini"  # bullish/bearish/neutral verdicts
    watch_news_max_age_hours: int = 24      # ignore articles older than this
    watch_alerts_retention: int = 500       # keep the newest N alert rows

    # Watchlists (configurable)
    # Emergency fallback only — universe.csv is the live source for Nordic tickers.
    # These are OMXS30 constituents in Yahoo Finance format (.ST not .STO).
    nordic_watchlist: list[str] = Field(
        default=[
            "ERIC-B.ST", "VOLV-B.ST", "SAND.ST", "SEB-A.ST", "SHB-A.ST",
            "SWED-A.ST", "AZN.ST", "INVE-B.ST", "ATCO-A.ST", "TELIA.ST",
            "ABB.ST", "ALFA.ST", "ALIV-SDB.ST", "ASSA-B.ST", "ATCO-B.ST",
            "BOL.ST", "EVO.ST", "GETI-B.ST", "HM-B.ST", "HEXA-B.ST",
            "HUSQ-B.ST", "KINV-B.ST", "LUND-B.ST", "NIBE-B.ST", "NDA-SE.ST",
            "SSAB-A.ST", "SKA-B.ST", "SKF-B.ST", "ESSITY-B.ST", "TEL2-B.ST",
        ]
    )
    us_watchlist: list[str] = Field(
        default=[
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B",
            "JPM", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "CVX", "MRK",
            "ABBV", "COST", "LLY", "AVGO", "PEP", "KO", "ADBE", "CRM", "WMT",
            "BAC", "TMO", "NFLX", "ACN", "AMD", "CSCO", "ABT", "DHR", "LIN",
            "INTC", "VZ", "CMCSA", "MCD", "TXN", "NEE", "PM", "RTX", "UPS",
        ]
    )

    # Paths (derived, not from env)
    @property
    def db_path(self) -> Path:
        return BASE_DIR / "data" / "deepswing.db"

    @property
    def heuristics_dir(self) -> Path:
        return BASE_DIR / "heuristics"

    @property
    def compiled_dir(self) -> Path:
        return BASE_DIR / "compiled"


settings = Settings()
