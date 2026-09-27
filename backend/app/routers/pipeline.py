from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.pipeline import (
    BatchPredictResponse,
    RetrainStatusResponse,
    RetrainHistoryResponse,
)
from app.utils.batch_predictor import run_batch_predictions
from app.models.retrain_history import RetrainHistory

router = APIRouter(prefix="/api/v1", tags=["pipeline"])


@router.post("/predictions/batch", response_model=BatchPredictResponse)
def run_batch(db: Session = Depends(get_db)):
    result = run_batch_predictions(db)
    return result


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
def trigger_retrain():
    from app.utils.scheduler import scheduled_retrain
    import threading

    thread = threading.Thread(target=scheduled_retrain, daemon=True)
    thread.start()
    return {"message": "Retrain triggered in background"}
