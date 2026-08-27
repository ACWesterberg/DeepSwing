from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from src.db import Base, Decision, _migrate_decisions

# Columns _migrate_decisions is responsible for backfilling onto databases that
# predate them. price/atr/entry_inputs came with counterfactual training; the
# last three came with the events track.
BACKFILLED = {"price", "atr", "entry_inputs", "fair_prob", "market_prob", "edge"}


def _legacy_engine(tmp_path):
    """A decisions table as it shipped before any of the backfilled columns."""
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE decisions (
                id INTEGER NOT NULL PRIMARY KEY,
                timestamp DATETIME,
                market VARCHAR(10) NOT NULL,
                track VARCHAR(10) NOT NULL,
                ticker VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,
                confidence FLOAT,
                rrr FLOAT,
                regime VARCHAR(20),
                reasoning TEXT,
                block_reason TEXT
            )
        """))
        conn.execute(text(
            "INSERT INTO decisions (market, track, ticker, action) "
            "VALUES ('us', 'claude', 'AAPL', 'BUY')"
        ))
    return engine


def _columns(engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("decisions")}


def test_backfills_every_added_column(tmp_path):
    engine = _legacy_engine(tmp_path)
    assert not BACKFILLED & _columns(engine)
    _migrate_decisions(engine)
    assert BACKFILLED <= _columns(engine)


def test_preserves_existing_rows(tmp_path):
    engine = _legacy_engine(tmp_path)
    _migrate_decisions(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker, fair_prob FROM decisions")).all()
    assert rows == [("AAPL", None)]


def test_is_idempotent(tmp_path):
    engine = _legacy_engine(tmp_path)
    _migrate_decisions(engine)
    _migrate_decisions(engine)
    columns = [c["name"] for c in inspect(engine).get_columns("decisions")]
    assert columns.count("fair_prob") == 1


def test_partial_migration_adds_only_what_is_missing(tmp_path):
    # A Pi that already ran the counterfactual migration but not the events one.
    engine = _legacy_engine(tmp_path)
    with engine.begin() as conn:
        for name in ("price", "atr"):
            conn.execute(text(f"ALTER TABLE decisions ADD COLUMN {name} FLOAT"))
    _migrate_decisions(engine)
    assert BACKFILLED <= _columns(engine)


def test_fresh_database_needs_no_migration(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(engine)
    _migrate_decisions(engine)
    assert BACKFILLED <= _columns(engine)


def test_event_decision_round_trips(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        session.add(Decision(
            market="events", track="claude_events", ticker="KXHIGHNY-26AUG28-B82",
            action="DRY_RUN", fair_prob=0.42, market_prob=0.30, edge=0.12,
        ))
        session.commit()
        stored = session.query(Decision).one()
        # "claude_events" is 13 chars — it needs the widened track column.
        assert stored.track == "claude_events"
        assert stored.to_dict()["edge"] == 0.12
    finally:
        session.close()
