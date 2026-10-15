from datetime import datetime
from pydantic import BaseModel, Field
from typing import Literal, Optional


class PredictRequest(BaseModel):
    customer_id: str = Field(..., max_length=50)
    threshold: float = Field(0.3, ge=0.0, le=1.0)


YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class PredictFromDataRequest(BaseModel):
    """Datos crudos de un cliente (no se guarda en la base de datos)."""
    gender: Literal["Male", "Female"]
    senior_citizen: bool
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(..., ge=0, le=72)
    phone_service: YesNo
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetAddon
    online_backup: InternetAddon
    device_protection: InternetAddon
    tech_support: InternetAddon
    streaming_tv: InternetAddon
    streaming_movies: InternetAddon
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
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
