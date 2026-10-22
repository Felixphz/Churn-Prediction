import sys
import os
import threading
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain import run_retrain

scheduler = BackgroundScheduler()
# Only one retrain at a time, whether it comes from the cron or from /trigger
_retrain_lock = threading.Lock()


def _run_locked():
    print(f"[{datetime.utcnow()}] Starting retrain...")
    try:
        run_retrain()
        print(f"[{datetime.utcnow()}] Retrain completed.")
    except Exception as e:
        print(f"[{datetime.utcnow()}] Retrain failed: {e}")
    finally:
        _retrain_lock.release()


def scheduled_retrain():
    if not _retrain_lock.acquire(blocking=False):
        print(f"[{datetime.utcnow()}] Retrain already running, skipping scheduled run.")
        return
    _run_locked()


def trigger_in_background() -> bool:
    """Start a retrain in a thread. Returns False if one is already running."""
    if not _retrain_lock.acquire(blocking=False):
        return False
    threading.Thread(target=_run_locked, daemon=True).start()
    return True


def is_running() -> bool:
    return _retrain_lock.locked()


def start_scheduler():
    scheduler.add_job(
        scheduled_retrain,
        CronTrigger(day=1, hour=2, minute=0),  # First day of month at 02:00
        id="monthly_retrain",
        name="Monthly model retrain",
        replace_existing=True,
    )
    scheduler.start()
    print("Scheduler started: monthly retrain on day 1 at 02:00")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        print("Scheduler stopped")
