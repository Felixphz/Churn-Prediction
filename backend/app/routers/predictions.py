from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.prediction import (
    PredictRequest,
    PredictionResponse,
    PredictionHistoryItem,
    PredictionListResponse,
)
from app.services.prediction_service import (
    predict_on_demand,
    get_prediction_history,
    get_latest_prediction,
)

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.post("/predict")
def run_prediction(request: PredictRequest, db: Session = Depends(get_db)):
    result = predict_on_demand(db, request.customer_id, request.threshold)
    if result is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return result


@router.get("/{customer_id}")
def get_latest(customer_id: str, db: Session = Depends(get_db)):
    prediction = get_latest_prediction(db, customer_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="No predictions found for this customer")
    return {
        "customer_id": customer_id,
        "prediction": {
            "churn_probability": float(prediction.churn_probability),
            "churn_predicted": prediction.churn_prediction,
            "threshold": float(prediction.threshold),
            "model": prediction.model_name,
            "model_version": prediction.model_version,
        },
        "explainability": {
            "top_features": prediction.top_features,
            "base_value": float(prediction.shap_base_value) if prediction.shap_base_value else None,
            "all_shap_values": prediction.shap_values,
        },
        "predicted_at": prediction.predicted_at.isoformat(),
        "prediction_type": prediction.prediction_type,
    }


@router.get("/{customer_id}/history", response_model=PredictionListResponse)
def get_history(customer_id: str, db: Session = Depends(get_db)):
    predictions, total = get_prediction_history(db, customer_id)
    return PredictionListResponse(predictions=predictions, total=total)
