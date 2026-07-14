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
    claude_erl_extended_thinking: bool = True
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
    drawdown_pause_threshold: float = 0.10
    max_sector_correlation: float = 0.7
    max_positions_per_sector: int = 2
    vix_halt_threshold: float = 35.0   # halt new entries when VIX >= this
    simulated_slippage: float = 0.0005     # 0.05% bid/ask spread approximation
    max_gap_slippage_pct: float = 0.02     # cap on extra fill slippage when price gaps through a stop
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
    rsi_min: float = 35.0                  # was 40.0
    rsi_max: float = 70.0                  # was 65.0
    volume_spike_multiplier: float = 1.2   # was 1.5 (20% above avg vol, not 50%)
    max_candidates_per_session: int = 15   # was 10
    earnings_buffer_days: int = 2          # exclude candidates within N days of earnings
    market_news_max_headlines: int = 20    # market-wide headlines injected into macro context

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
    # Holdings are monitored on price alone; a news pull + AI exit review only fires
    # for a position once it has moved at least this fraction (up or down) since its
    # last news check — a "large jump". Set to 0.0 to review every scan.
    holdings_news_jump_pct: float = 0.05

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
