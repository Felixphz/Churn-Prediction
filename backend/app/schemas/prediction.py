from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    customer_id: str = Field(..., max_length=50)
    threshold: float = Field(0.3, ge=0.0, le=1.0)


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    customer_id: str
    prediction: dict
    explainability: dict
    predicted_at: datetime
    prediction_type: str


class PredictionHistoryItem(BaseModel):
    id: int
    model_name: str
    model_version: Optional[str]
    churn_probability: float
    churn_prediction: bool
    prediction_type: str
    predicted_at: datetime

    class Config:
        from_attributes = True


class PredictionListResponse(BaseModel):
    predictions: list[PredictionHistoryItem]
    total: int
