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

    # Options tracks — long single-leg US calls only (v1); empty list disables
    options_tracks: list[Literal["claude-opt", "gpt-opt"]] = ["claude-opt", "gpt-opt"]
    options_starting_capital_sek: float = 100_000.0

    @property
    def all_tracks(self) -> list[str]:
        return [*self.tracks, *self.options_tracks]

    # Claude models
    claude_decision_model: str = "claude-sonnet-5"          # scan decisions (up from Haiku)
    claude_erl_model: str = "claude-opus-4-8"               # heavy post-trade reasoning
    claude_erl_extended_thinking: bool = True               # adaptive thinking on Opus 4.8
    claude_erl_effort: str = "high"                         # output_config.effort; low|medium|high|max
    claude_prompt_model: str = "claude-opus-4-8"            # MIPRO instruction proposer

    # GPT models
    gpt_decision_model: str = "gpt-5"                        # scan decisions (up from 4o-mini)
    gpt_news_model: str = "gpt-5-mini"                       # shared news analysis (light task)
    gpt_erl_model: str = "gpt-5.5"                           # heavy post-trade reasoning
    gpt_erl_reasoning_effort: str = "high"                   # GPT "thinking" for ERL; "" disables
    gpt_prompt_model: str = "gpt-5.5"                        # MIPRO instruction proposer

    # Risk parameters
    max_risk_per_trade: float = 0.01       # 1% of portfolio
    hard_cap_risk_per_trade: float = 0.02  # 2% hard cap
    min_rrr: float = 2.0
    atr_stop_multiplier: float = 1.5
    # Risk-based sizing alone is unbounded (tight stop → huge position), so position
    # value is also capped as a fraction of equity — and at available cash.
    max_position_pct: float = 0.25
    # Trailing stop distance in ATRs once a position is in profit. Wider than the
    # entry stop (1.5×ATR) so ordinary daily noise doesn't knock out winners
    # before the RRR>=2 target is reachable.
    trailing_stop_atr_multiplier: float = 2.0
    drawdown_pause_threshold: float = 0.10
    max_sector_correlation: float = 0.7
    max_positions_per_sector: int = 2
    vix_halt_threshold: float = 35.0   # halt new entries when VIX >= this
    simulated_slippage: float = 0.0005     # 0.05% bid/ask spread approximation
    # Montrose Premium: 0.10% courtage each way; 0.10% FX fee for non-SEK trades
    commission_pct: float = 0.001          # 0.10% per trade leg (buy + sell)
    fx_commission_pct: float = 0.001       # 0.10% extra on USD/EUR legs (US market)

    # Dashboard security
    reset_pin: str = "3821"
    dashboard_user: str = "deepswing"
    dashboard_password: str = ""   # leave empty to disable auth

    # Screener thresholds — loosened to widen the funnel (more at-bats for MIPRO
    # to learn from). The AI decision + RRR>=2 risk validation remain the quality
    # gate downstream, so this raises trade volume without lowering standards.
    # Hurst estimation basis. R/S on price *levels* (legacy default) biases H
    # upward — a plain drifting random walk reads "trending". On *returns* H
    # measures persistence properly, but drifting walks then read ~0.5 (neutral)
    # and the screener gets much stricter. Flip deliberately and observe.
    hurst_on_returns: bool = False

    rsi_min: float = 35.0                  # was 40.0
    rsi_max: float = 70.0                  # was 65.0
    volume_spike_multiplier: float = 1.2   # was 1.5 (20% above avg vol, not 50%)
    max_candidates_per_session: int = 15   # was 10
    earnings_buffer_days: int = 2          # exclude candidates within N days of earnings
    market_news_max_headlines: int = 20    # market-wide headlines injected into macro context

    # yfinance batch downloads fail above ~200 symbols — chunk large universe
    # watchlists so cold-cache scans still populate every ticker.
    ohlcv_batch_chunk_size: int = 150

    # Scheduler intervals (minutes)
    scan_interval_minutes: int = 15
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
    counterfactual_max_examples: int = 30        # cap so counterfactuals can't swamp real trades

    # Housekeeping: decisions accumulate ~1k rows/day at 15-min scans; prune rows
    # older than this during weekly maintenance (0 disables). Counterfactual
    # training only reads recent PASSes, so 90 days is generous.
    decisions_retention_days: int = 90
    # Daily on-disk SQLite snapshot (data/backups/), keep the newest N (0 disables).
    # MIPRO programs are backed up offsite; this protects the portfolio DB itself.
    db_backup_keep: int = 7

    # Options track — chain shortlist filters (yfinance US chains only)
    options_min_dte: int = 21
    options_max_dte: int = 60
    options_expiries_considered: int = 2   # nearest expiries inside the DTE window
    options_delta_min: float = 0.35
    options_delta_max: float = 0.65
    options_min_open_interest: int = 200
    options_min_volume: int = 10
    options_max_spread_pct: float = 0.08   # (ask-bid)/mid ceiling
    options_shortlist_size: int = 8
    options_risk_free_rate: float = 0.04   # Black-Scholes r for local greeks

    # Options track — risk (max loss on a long option IS the premium paid)
    options_max_premium_pct: float = 0.01       # premium budget per trade, % of equity
    options_hard_cap_premium_pct: float = 0.02  # one contract may stretch to this
    options_profit_target_bounds: tuple[float, float] = (0.20, 3.00)  # +% of premium
    options_max_loss_bounds: tuple[float, float] = (0.30, 0.60)       # -% of premium
    options_time_stop_min_dte: int = 7          # force-close at this DTE at the latest
    options_commission_per_contract_sek: float = 15.0  # flat per contract, each way

    # Options track — scheduling (hourly: chains are slow + decisions cost LLM calls)
    options_scan_interval_minutes: int = 60

    # MIPRO artifact backup — path to a local git working copy of a standalone
    # backups repo (e.g. ~/Github/deepswing-mipro-backups). Set via env
    # MIPRO_BACKUP_REPO_DIR. Empty disables backup. The Pi must have push
    # credentials configured on that working copy's remote.
    mipro_backup_repo_dir: str = ""
    mipro_backup_push: bool = True  # commit locally always; push to remote if True

    # Boot-time preflight: ping each configured model once so a bad ID/credential
    # surfaces immediately in the logs instead of at the next scan/ERL/MIPRO run.
    preflight_check_models: bool = True

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
