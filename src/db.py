from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings


class Base(DeclarativeBase):
    pass


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    track = Column(String(10), nullable=False)  # "claude" | "gpt"
    ticker = Column(String(20), nullable=False)
    market = Column(String(10), nullable=False)  # "nordic" | "us"
    action = Column(String(4), nullable=False)   # "BUY" | "SELL"
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    regime = Column(String(20))
    reasoning = Column(Text)
    confidence = Column(Float)
    pnl = Column(Float)
    pnl_pct = Column(Float)
    rrr_achieved = Column(Float)
    is_open = Column(Boolean, default=True)
    signals = Column(JSON)

    __table_args__ = (
        Index("ix_trades_track_open", "track", "is_open"),
        Index("ix_trades_ticker", "ticker"),
    )


class Position(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True)
    trade_id = Column(Integer, nullable=False)
    track = Column(String(10), nullable=False)
    ticker = Column(String(20), nullable=False)
    market = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    stop_loss = Column(Float, nullable=False)
    target = Column(Float, nullable=False)
    trailing_stop = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_positions_track", "track"),)


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True)
    track = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    open_positions_value = Column(Float, nullable=False)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    avg_rrr = Column(Float)

    __table_args__ = (Index("ix_snapshots_track_time", "track", "timestamp"),)


class PortfolioState(Base):
    """Full live state of one track's portfolio — the durable mirror of the
    in-memory Portfolio, so tracks survive a process restart / redeploy."""
    __tablename__ = "portfolio_state"

    track = Column(String(10), primary_key=True)
    updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cash = Column(Float, nullable=False)
    starting_equity = Column(Float, nullable=False)
    peak_equity = Column(Float, nullable=False)
    total_commission = Column(Float, default=0.0)
    next_trade_id = Column(Integer, default=1)
    open_positions = Column(JSON, default=list)   # list of serialized OpenPosition
    closed_trades = Column(JSON, default=list)     # list of serialized ClosedTrade


class Heuristic(Base):
    __tablename__ = "heuristics"

    id = Column(String(36), primary_key=True)  # UUID
    track = Column(String(10), nullable=False)
    trigger = Column(Text, nullable=False)
    action = Column(Text, nullable=False)
    market = Column(String(10), default="both")
    regime = Column(String(20), default="any")
    quality_score = Column(Float, default=5.0)
    access_count = Column(Integer, default=0)
    is_core = Column(Boolean, default=False)
    created = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime)
    source_trade_id = Column(Integer)

    __table_args__ = (
        Index("ix_heuristics_track_quality", "track", "quality_score"),
    )


class Decision(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    market = Column(String(10), nullable=False)   # "nordic" | "us"
    track = Column(String(10), nullable=False)    # "claude" | "gpt"
    ticker = Column(String(20), nullable=False)
    action = Column(String(10), nullable=False)   # BUY | HOLD | SELL | BLOCKED | ERROR
    confidence = Column(Float)
    rrr = Column(Float)
    regime = Column(String(20))
    reasoning = Column(Text)
    block_reason = Column(Text)

    __table_args__ = (
        Index("ix_decisions_time", "timestamp"),
        Index("ix_decisions_ticker", "ticker"),
        Index("ix_decisions_track_time", "track", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "market": self.market,
            "track": self.track,
            "ticker": self.ticker,
            "action": self.action,
            "confidence": self.confidence,
            "rrr": self.rrr,
            "regime": self.regime,
            "reasoning": self.reasoning,
            "reason": self.block_reason,
        }


def get_engine():
    db_url = f"sqlite:///{settings.db_path}"
    return create_engine(db_url, connect_args={"check_same_thread": False})


def init_db() -> None:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()
