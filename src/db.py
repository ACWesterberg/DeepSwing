from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
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


# NOTE: earlier schema revisions defined Trade, Position, PortfolioSnapshot and
# Heuristic tables here — none were ever written. Live state is the
# portfolio_state row per track; heuristics are file-backed; decisions are the
# audit trail. The model classes are gone; empty tables in existing DBs on the
# Pi are harmless leftovers.


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
    # Counterfactual training support: decision-time price + ATR (native currency)
    # and the exact DSPy inputs. Only populated for the first PASS per
    # track/ticker/day so the optimizer can label passed-on setups from
    # subsequent price data (ATR lets it simulate the stop/target path).
    price = Column(Float)
    atr = Column(Float)
    # none_as_null: Python None must become SQL NULL (not the JSON string 'null')
    # or the isnot(None) filters in the counterfactual builder match empty rows
    entry_inputs = Column(JSON(none_as_null=True))

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
    _migrate_decisions(engine)


def _migrate_decisions(engine) -> None:
    """create_all never alters existing tables — add columns introduced after
    the decisions table first shipped (SQLite supports ADD COLUMN)."""
    from sqlalchemy import text

    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(decisions)"))}
        for name, ddl in (("price", "FLOAT"), ("atr", "FLOAT"), ("entry_inputs", "JSON")):
            if name not in existing:
                conn.execute(text(f"ALTER TABLE decisions ADD COLUMN {name} {ddl}"))
        conn.commit()


def get_session() -> Session:
    engine = get_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()


def prune_old_decisions(retention_days: int) -> int:
    """Delete decision rows older than retention_days (0 disables). Returns count."""
    if retention_days <= 0:
        return 0
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(days=retention_days)
    session = get_session()
    try:
        deleted = (
            session.query(Decision)
            .filter(Decision.timestamp < cutoff)
            .delete(synchronize_session=False)
        )
        session.commit()
        return deleted
    finally:
        session.close()
