from __future__ import annotations

import logging
import sys

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import settings
from src.db import init_db
from src.scheduler.market_hours import is_market_open
from src.scheduler.markets import SCAN_MARKETS
from src.scheduler.optimizer import run_heuristic_refinement, run_mipro_optimization, run_options_mipro
from src.scheduler.options_scan import run_expiry_sweep, run_options_scan
from src.scheduler.scan_loop import run_scan

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("deepswing")


def scheduled_scan():
    """Called every 15 min by APScheduler — runs scan for any open market."""
    for market in SCAN_MARKETS:
        if is_market_open(market):
            try:
                run_scan(market)
            except Exception as exc:
                logger.error("Scan error for %s: %s", market, exc, exc_info=True)


def scheduled_options_scan():
    """Called hourly by APScheduler — options ride the US session only."""
    if not settings.options_tracks or not is_market_open("us"):
        return
    try:
        run_options_scan()
    except Exception as exc:
        logger.error("Options scan error: %s", exc, exc_info=True)


def scheduled_watch_monitor():
    """Personal watchlist: move/news/insider alerts to Telegram. News and insider
    checks run around the clock (cached fetches keep it cheap); the move check
    gates itself on market hours inside the monitor."""
    from src.scheduler.watch_monitor import run_watch_monitor
    try:
        run_watch_monitor()
    except Exception as exc:
        logger.error("Watch monitor error: %s", exc, exc_info=True)


def scheduled_expiry_sweep():
    """Daily post-US-close settlement of expired option contracts."""
    if not settings.options_tracks:
        return
    try:
        run_expiry_sweep()
    except Exception as exc:
        logger.error("Expiry sweep error: %s", exc, exc_info=True)


def weekly_maintenance():
    """Run MIPRO optimization + heuristic refinement for all tracks."""
    for track in settings.tracks:
        try:
            run_heuristic_refinement(track)
            run_mipro_optimization(track)
        except Exception as exc:
            logger.error("Weekly maintenance error for %s: %s", track, exc, exc_info=True)
    for track in settings.options_tracks:
        try:
            run_heuristic_refinement(track)
            run_options_mipro(track)
        except Exception as exc:
            logger.error("Weekly maintenance error for %s: %s", track, exc, exc_info=True)
    try:
        from src.db import prune_old_decisions
        pruned = prune_old_decisions(settings.decisions_retention_days)
        if pruned:
            logger.info("Pruned %d decision rows older than %d days", pruned, settings.decisions_retention_days)
    except Exception as exc:
        logger.error("Decision retention pruning error: %s", exc)


def daily_db_backup():
    """Nightly on-disk SQLite snapshot with rotation."""
    from src.scheduler.backup import backup_database
    backup_database()


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="Europe/Stockholm")

    # Every 15 minutes — market-hours-aware scan
    scheduler.add_job(
        scheduled_scan,
        "interval",
        minutes=settings.scan_interval_minutes,
        id="market_scan",
        max_instances=1,
        coalesce=True,
    )

    # Options scan (hourly by default) + daily expiry sweep after US close
    if settings.options_tracks:
        scheduler.add_job(
            scheduled_options_scan,
            "interval",
            minutes=settings.options_scan_interval_minutes,
            id="options_scan",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            scheduled_expiry_sweep,
            "cron",
            day_of_week="mon-fri",
            hour=22,
            minute=10,
            id="options_expiry_sweep",
            max_instances=1,
        )

    # Personal watchlist monitor — light (one quote + cached news per ticker),
    # so it never takes the scan lock and can't be starved by a long scan
    scheduler.add_job(
        scheduled_watch_monitor,
        "interval",
        minutes=settings.watch_interval_minutes,
        id="watch_monitor",
        max_instances=1,
        coalesce=True,
    )

    # Weekly Sunday at 02:00 CET — MIPRO + heuristic maintenance
    scheduler.add_job(
        weekly_maintenance,
        "cron",
        day_of_week="sun",
        hour=2,
        minute=0,
        id="weekly_maintenance",
        max_instances=1,
    )

    # Nightly 23:45 CET — SQLite snapshot (after both markets close)
    scheduler.add_job(
        daily_db_backup,
        "cron",
        hour=23,
        minute=45,
        id="db_backup",
        max_instances=1,
    )

    scheduler.start()
    logger.info("Scheduler started (scan every %dm, weekly maintenance Sunday 02:00 CET)", settings.scan_interval_minutes)
    return scheduler


def main():
    logger.info("DeepSwing starting up...")

    # Initialize database
    init_db()
    logger.info("Database initialized at %s", settings.db_path)

    # Restore persisted portfolio state, then arm persistence so tracks survive
    # future restarts. Restore runs with the handler off, so it never writes back.
    from src.portfolio.persistence import restore_portfolios, save_portfolio
    from src.portfolio.simulator import set_persistence_handler
    restore_portfolios()
    set_persistence_handler(save_portfolio)

    # Ensure compiled + heuristic dirs exist
    settings.compiled_dir.mkdir(parents=True, exist_ok=True)
    for track in settings.all_tracks:
        (settings.heuristics_dir / track).mkdir(parents=True, exist_ok=True)

    logger.info("Simulation tracks: %s", settings.all_tracks)

    # Log resolved model IDs, and optionally ping each so a bad ID/credential
    # surfaces now rather than at the next scan/ERL/MIPRO run.
    from src.scheduler.preflight import check_models, log_model_config
    log_model_config()
    if settings.preflight_check_models:
        check_models()

    # Start background scheduler
    scheduler = start_scheduler()

    # Start FastAPI + uvicorn
    from src.dashboard.app import app
    try:
        uvicorn.run(
            app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level=settings.log_level.lower(),
        )
    finally:
        scheduler.shutdown()
        logger.info("DeepSwing stopped.")


if __name__ == "__main__":
    main()
