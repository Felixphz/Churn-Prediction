from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class BatchPredictResponse(BaseModel):
    batch_run_id: str
    total_customers: int
    churn_predicted: int
    churn_rate: float
    errors: int
    started_at: str
    finished_at: str
    duration_seconds: float


class RetrainStatusResponse(BaseModel):
    id: int
    run_id: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime]
    models_evaluated: Optional[int]
    best_model_name: Optional[str]
    best_f1_score: Optional[float]
    best_recall: Optional[float]
    best_auc_roc: Optional[float]
    model_improved: bool
    error_message: Optional[str]

    class Config:
        from_attributes = True


class RetrainHistoryResponse(BaseModel):
    runs: list[RetrainStatusResponse]
    total: int


class DashboardSummary(BaseModel):
    total_customers: int
    churn_count: int
    churn_rate: float
    active_model: Optional[str]
    total_predictions: int
    last_retrain: Optional[str]
