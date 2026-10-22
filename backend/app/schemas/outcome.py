from datetime import date, datetime
from pydantic import BaseModel, Field
from typing import Optional


class OutcomeCreate(BaseModel):
    customer_id: str = Field(..., max_length=50)
    churned: bool
    retention_action: bool = False  # a retention plan was applied before the outcome
    observed_at: date = Field(default_factory=date.today)


class OutcomeResponse(BaseModel):
    id: int
    customer_id: str
    churn: bool
    retention_action: bool
    label_source: str
    observed_at: date
    snapshot_at: datetime
    prediction_id: Optional[int]
    predicted_probability: Optional[float]
    model_version: Optional[str]
    customer_active: bool

    class Config:
        from_attributes = True


class OutcomeSummaryResponse(BaseModel):
    total_samples: int
    by_source: dict[str, int]
    churn_rate: float
    treated: int
    treated_retention_rate: Optional[float]  # % of treated customers that stayed
    trainable_samples: int  # what the next retrain will use
