from __future__ import annotations

import logging
from datetime import datetime

from config.settings import settings
from src.db import PortfolioState, get_session
from src.portfolio.simulator import get_portfolio

logger = logging.getLogger(__name__)


def save_portfolio(portfolio) -> None:
    """Mirror a track's (stock or options) full live state to the DB. Best-effort — never raises."""
    state = portfolio.export_state()
    session = get_session()
    try:
        row = session.get(PortfolioState, portfolio.track)
        if row is None:
            row = PortfolioState(track=portfolio.track)
            session.add(row)
        row.cash = state["cash"]
        row.starting_equity = state["starting_equity"]
        row.peak_equity = state["peak_equity"]
        row.total_commission = state["total_commission"]
        row.next_trade_id = state["next_trade_id"]
        row.open_positions = state["open_positions"]
        row.closed_trades = state["closed_trades"]
        row.updated = datetime.utcnow()
        session.commit()
    finally:
        session.close()


def restore_portfolios() -> None:
    """Rehydrate in-memory portfolios from the DB on startup. Missing/blank rows
    leave a track at its fresh starting capital."""
    from src.portfolio.options_simulator import get_options_portfolio

    session = get_session()
    try:
        for track in settings.all_tracks:
            row = session.get(PortfolioState, track)
            if row is None:
                continue
            if track in settings.options_tracks:
                portfolio = get_options_portfolio(track)
                # An options track that never traded carries no history worth
                # keeping — rebase it to the current starting capital instead of
                # resurrecting the old (too-small) bankroll it was created with.
                if (
                    not (row.open_positions or row.closed_trades)
                    and row.starting_equity != settings.options_starting_capital_sek
                ):
                    logger.info(
                        "Rebasing untraded %s from %.0f to %.0f SEK starting capital",
                        track, row.starting_equity, settings.options_starting_capital_sek,
                    )
                    continue
            else:
                portfolio = get_portfolio(track)
            portfolio.import_state({
                "cash": row.cash,
                "starting_equity": row.starting_equity,
                "peak_equity": row.peak_equity,
                "total_commission": row.total_commission,
                "next_trade_id": row.next_trade_id,
                "open_positions": row.open_positions or [],
                "closed_trades": row.closed_trades or [],
            })
            logger.info(
                "Restored %s portfolio: cash=%.2f, equity=%.2f, %d open, %d closed",
                track, portfolio.cash, portfolio.equity,
                len(portfolio.open_positions), len(portfolio.closed_trades),
            )
    except Exception as exc:
        logger.warning("Portfolio restore failed (tracks start fresh): %s", exc)
    finally:
        session.close()


def delete_portfolio_state(tracks: list[str]) -> None:
    """Remove persisted state for the given tracks (used by /api/reset)."""
    session = get_session()
    try:
        session.query(PortfolioState).filter(PortfolioState.track.in_(tracks)).delete(
            synchronize_session=False
        )
        session.commit()
    finally:
        session.close()
