from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from app.schemas.pipeline import DashboardSummary

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(db: Session = Depends(get_db)):
    total_customers = db.query(func.count(Customer.id)).scalar() or 0
    total_predictions = db.query(func.count(Prediction.id)).scalar() or 0

    churn_count = (
        db.query(func.count(Prediction.id))
        .filter(Prediction.churn_prediction == True)
        .scalar()
        or 0
    )

    active_model = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()

    last_retrain = (
        db.query(RetrainHistory)
        .order_by(RetrainHistory.started_at.desc())
        .first()
    )

    churn_rate = round(churn_count / total_customers * 100, 2) if total_customers > 0 else 0

    return DashboardSummary(
        total_customers=total_customers,
        churn_count=churn_count,
        churn_rate=churn_rate,
        active_model=active_model.model_name if active_model else None,
        total_predictions=total_predictions,
        last_retrain=last_retrain.started_at.isoformat() if last_retrain else None,
    )
