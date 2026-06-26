from __future__ import annotations

import logging
import sys

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import settings
from src.db import init_db
from src.scheduler.market_hours import is_market_open
from src.scheduler.optimizer import run_heuristic_refinement, run_mipro_optimization
from src.scheduler.scan_loop import run_scan

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("deepswing")


def scheduled_scan():
    """Called every 15 min by APScheduler — runs scan for any open market."""
    for market in ("nordic", "us"):
        if is_market_open(market):
            try:
                run_scan(market)
            except Exception as exc:
                logger.error("Scan error for %s: %s", market, exc, exc_info=True)


def weekly_maintenance():
    """Run MIPRO optimization + heuristic refinement for all tracks."""
    for track in settings.tracks:
        try:
            run_heuristic_refinement(track)
            run_mipro_optimization(track)
        except Exception as exc:
            logger.error("Weekly maintenance error for %s: %s", track, exc, exc_info=True)


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

    scheduler.start()
    logger.info("Scheduler started (scan every %dm, weekly maintenance Sunday 02:00 CET)", settings.scan_interval_minutes)
    return scheduler


def main():
    logger.info("DeepSwing starting up...")

    # Initialize database
    init_db()
    logger.info("Database initialized at %s", settings.db_path)

    # Ensure compiled + heuristic dirs exist
    settings.compiled_dir.mkdir(parents=True, exist_ok=True)
    for track in settings.tracks:
        (settings.heuristics_dir / track).mkdir(parents=True, exist_ok=True)

    logger.info("Simulation tracks: %s", settings.tracks)

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
