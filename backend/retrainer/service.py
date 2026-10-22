"""Retrainer service for local docker-compose: monthly cron + manual trigger.

The API proxies /api/v1/retrain/trigger here when GITHUB_TOKEN is not set.
In production the retrain runs on GitHub Actions (.github/workflows/monthly-retrain.yml).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from retrainer.scheduler import is_running, start_scheduler, stop_scheduler, trigger_in_background

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting Churn Retrainer - {settings.ENVIRONMENT}")
    if settings.ENVIRONMENT != "test" and settings.ENABLE_SCHEDULER:
        start_scheduler()
    yield
    stop_scheduler()
    print("Shutting down Churn Retrainer")


app = FastAPI(title="Churn Retrainer", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy", "retrain_running": is_running()}


@app.post("/trigger")
def trigger():
    if not trigger_in_background():
        raise HTTPException(status_code=409, detail="A retrain is already running")
    return {"message": "Retrain triggered in background"}

