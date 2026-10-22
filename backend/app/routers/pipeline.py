from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.config import get_settings
from app.database import get_db
from app.schemas.pipeline import (
    BatchPredictResponse,
    RetrainStatusResponse,
    RetrainHistoryResponse,
)
from sqlalchemy import func
from app.utils.batch_predictor import BATCH_RUNS, start_batch_predictions
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.retrain_history import RetrainHistory

settings = get_settings()
router = APIRouter(prefix="/api/v1", tags=["pipeline"])


def _dispatch_github_workflow():
    """Production: run the retrain workflow on GitHub Actions (workflow_dispatch)."""
    url = (
        f"https://api.github.com/repos/{settings.GITHUB_REPO}"
        f"/actions/workflows/{settings.RETRAIN_WORKFLOW}/dispatches"
    )
    try:
        res = httpx.post(
            url,
            json={"ref": "main"},
            headers={
                "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {e}")
    if res.status_code != 204:
        raise HTTPException(status_code=502, detail=f"GitHub workflow dispatch failed: {res.text}")


def _trigger_retrainer_service():
    """Local (docker-compose): forward to the retrainer service, passing its errors through."""
    try:
        res = httpx.post(f"{settings.RETRAIN_SERVICE_URL}/trigger", timeout=30)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Retrainer service unavailable: {e}")
    if res.status_code >= 400:
        raise HTTPException(status_code=res.status_code, detail=res.json().get("detail", res.text))


@router.post("/predictions/batch", response_model=BatchPredictResponse, status_code=202)
def run_batch():
    """Lanza el batch en segundo plano; consultar el avance en GET /predictions/batch/{id}."""
    return start_batch_predictions()


@router.get("/predictions/batch/{batch_run_id}", response_model=BatchPredictResponse)
def batch_status(batch_run_id: str, db: Session = Depends(get_db)):
    if batch_run_id in BATCH_RUNS:
        return BATCH_RUNS[batch_run_id]

    # Batch de un proceso anterior (p. ej. tras un reinicio): reconstruir desde la BD
    processed, churn = (
        db.query(
            func.count(Prediction.id),
            func.count(Prediction.id).filter(Prediction.churn_prediction == True),
        )
        .filter(Prediction.batch_run_id == batch_run_id)
        .one()
    )
    if not processed:
        raise HTTPException(status_code=404, detail="Batch run not found")
    total = db.query(func.count(Customer.id)).filter(Customer.active == True).scalar() or 0
    return {
        "batch_run_id": batch_run_id,
        "status": "completed" if processed >= total else "incomplete",
        "total_customers": total,
        "processed": processed,
        "churn_predicted": churn,
        "churn_rate": round(churn / processed * 100, 2),
        "errors": 0,
    }


@router.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status(db: Session = Depends(get_db)):
    latest = (
        db.query(RetrainHistory)
        .order_by(RetrainHistory.started_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No retrain history found")
    return latest


@router.get("/retrain/history", response_model=RetrainHistoryResponse)
def retrain_history(
    limit: int = 10,
    db: Session = Depends(get_db),
):
    runs = (
        db.query(RetrainHistory)
        .order_by(RetrainHistory.started_at.desc())
        .limit(limit)
        .all()
    )
    return RetrainHistoryResponse(runs=runs, total=len(runs))


@router.post("/retrain/trigger")
def trigger_retrain(db: Session = Depends(get_db)):
    latest = db.query(RetrainHistory).order_by(RetrainHistory.started_at.desc()).first()
    stale_after = timedelta(hours=settings.RETRAIN_STALE_HOURS)
    if latest and latest.status == "running" and datetime.utcnow() - latest.started_at < stale_after:
        raise HTTPException(status_code=409, detail="A retrain is already running")

    if settings.GITHUB_TOKEN and settings.GITHUB_REPO:
        _dispatch_github_workflow()
    else:
        _trigger_retrainer_service()
    return {"message": "Retrain triggered in background"}


@router.get("/retrain/top-models")
def retrain_top_models(limit: int | None = None):
    """Best MLflow runs by CV F1-macro, kept for analysis after each retrain."""
    from scripts.retrain import top_models

    try:
        models = top_models(limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {e}")
    return {"models": models, "total": len(models)}
