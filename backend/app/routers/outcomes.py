from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.config import get_settings
from app.database import get_db
from app.training_database import get_training_db
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.training_sample import FEATURE_FIELDS, TrainingSample
from app.schemas.outcome import OutcomeCreate, OutcomeResponse, OutcomeSummaryResponse

settings = get_settings()
router = APIRouter(prefix="/api/v1/outcomes", tags=["outcomes"])


@router.post("", response_model=OutcomeResponse, status_code=201)
def record_outcome(
    outcome: OutcomeCreate,
    db: Session = Depends(get_db),
    training_db: Session = Depends(get_training_db),
):
    """Record a confirmed outcome: snapshot the customer into the training DB."""
    customer = db.query(Customer).filter(Customer.customer_id == outcome.customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Latest prediction is kept for predicted-vs-actual monitoring only
    last = (
        db.query(Prediction)
        .filter(Prediction.customer_id == customer.id)
        .order_by(Prediction.predicted_at.desc())
        .first()
    )
    sample = TrainingSample(
        customer_id=customer.customer_id,
        **{field: getattr(customer, field) for field in FEATURE_FIELDS},
        churn=outcome.churned,
        retention_action=outcome.retention_action,
        label_source="observed",
        observed_at=outcome.observed_at,
        prediction_id=last.id if last else None,
        predicted_probability=last.churn_probability if last else None,
        model_version=last.model_version if last else None,
    )
    training_db.add(sample)
    try:
        training_db.commit()
    except IntegrityError:
        training_db.rollback()
        raise HTTPException(status_code=409, detail="Outcome already recorded for this customer and date")
    training_db.refresh(sample)

    # Operational DB only changes once the training sample is stored
    if outcome.churned and customer.active:
        customer.active = False
        db.commit()

    return OutcomeResponse(
        **{c: getattr(sample, c) for c in OutcomeResponse.model_fields if c != "customer_active"},
        customer_active=customer.active,
    )


@router.get("/summary", response_model=OutcomeSummaryResponse)
def outcomes_summary(training_db: Session = Depends(get_training_db)):
    by_source = dict(
        training_db.query(TrainingSample.label_source, func.count(TrainingSample.id))
        .group_by(TrainingSample.label_source)
        .all()
    )
    total = sum(by_source.values())
    churned = training_db.query(func.count(TrainingSample.id)).filter(TrainingSample.churn == True).scalar()
    treated, treated_stayed = (
        training_db.query(
            func.count(TrainingSample.id),
            func.count(TrainingSample.id).filter(TrainingSample.churn == False),
        )
        .filter(TrainingSample.retention_action == True)
        .one()
    )
    return OutcomeSummaryResponse(
        total_samples=total,
        by_source=by_source,
        churn_rate=round(churned / total * 100, 2) if total else 0.0,
        treated=treated,
        treated_retention_rate=round(treated_stayed / treated * 100, 2) if treated else None,
        trainable_samples=total if settings.RETRAIN_INCLUDE_TREATED else total - treated,
    )
