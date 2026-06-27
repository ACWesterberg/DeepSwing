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
    fred_api_key: str = ""

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    log_level: str = "INFO"

    # Simulation tracks
    tracks: list[Literal["claude", "gpt"]] = ["claude", "gpt"]
    starting_capital_sek: float = 100_000.0

    # Claude models
    claude_decision_model: str = "claude-haiku-4-5"
    claude_erl_model: str = "claude-sonnet-4-6"
    claude_erl_extended_thinking: bool = True

    # GPT models
    gpt_decision_model: str = "gpt-4o-mini"
    gpt_erl_model: str = "gpt-4o"

    # Risk parameters
    max_risk_per_trade: float = 0.01       # 1% of portfolio
    hard_cap_risk_per_trade: float = 0.02  # 2% hard cap
    min_rrr: float = 2.0
    atr_stop_multiplier: float = 1.5
    drawdown_pause_threshold: float = 0.10
    max_sector_correlation: float = 0.7
    max_positions_per_sector: int = 2
    vix_halt_threshold: float = 35.0   # halt new entries when VIX >= this
    simulated_slippage: float = 0.0005     # 0.05%

    # Screener thresholds
    rsi_min: float = 40.0
    rsi_max: float = 65.0
    volume_spike_multiplier: float = 1.5
    max_candidates_per_session: int = 10

    # Scheduler intervals (minutes)
    scan_interval_minutes: int = 15
    news_refresh_interval_minutes: int = 60

    # Watchlists (configurable)
    nordic_watchlist: list[str] = Field(
        default=[
            "ERIC-B.STO", "VOLV-B.STO", "SAND.STO", "SEB-A.STO", "SHB-A.STO",
            "SWED-A.STO", "AZN.STO", "INVE-B.STO", "ATCO-A.STO", "TELIA.STO",
            "ABB.STO", "ALFA.STO", "ALIV-SDB.STO", "ASSA-B.STO", "ATCO-B.STO",
            "BOL.STO", "EVO.STO", "GETI-B.STO", "HM-B.STO", "HEXA-B.STO",
            "HUSQ-B.STO", "KINV-B.STO", "LUND-B.STO", "NIBE-B.STO", "NDA-SE.STO",
            "SSAB-A.STO", "SKA-B.STO", "SKF-B.STO", "ESSITY-B.STO", "TEL2-B.STO",
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
