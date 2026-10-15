from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
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

router = APIRouter(prefix="/api/v1", tags=["pipeline"])


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
    total = db.query(func.count(Customer.id)).scalar() or 0
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
def trigger_retrain():
    from app.utils.scheduler import scheduled_retrain
    import threading

    thread = threading.Thread(target=scheduled_retrain, daemon=True)
    thread.start()
    return {"message": "Retrain triggered in background"}
