import sys
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain import run_retrain

scheduler = BackgroundScheduler()


def scheduled_retrain():
    print(f"[{datetime.utcnow()}] Starting scheduled retrain...")
    try:
        run_retrain()
        print(f"[{datetime.utcnow()}] Scheduled retrain completed.")
    except Exception as e:
        print(f"[{datetime.utcnow()}] Scheduled retrain failed: {e}")


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
