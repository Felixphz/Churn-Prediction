from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Numeric, Date, DateTime, UniqueConstraint
)
from app.training_database import TrainingBase

# Feature columns shared with Customer (snapshot copied on each outcome)
FEATURE_FIELDS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure", "phone_service",
    "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method", "monthly_charges", "total_charges",
]


class TrainingSample(TrainingBase):
    """Customer feature snapshot + observed outcome. Lives in the training DB."""
    __tablename__ = "training_samples"
    __table_args__ = (UniqueConstraint("customer_id", "observed_at", name="uq_training_customer_observed"),)

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String(50), nullable=False, index=True)  # external id, no FK (other DB)

    # Features (same names/types as Customer)
    gender = Column(String(10), nullable=False)
    senior_citizen = Column(Boolean, nullable=False)
    partner = Column(String(5), nullable=False)
    dependents = Column(String(5), nullable=False)
    tenure = Column(Integer, nullable=False)
    phone_service = Column(String(5), nullable=False)
    multiple_lines = Column(String(20))
    internet_service = Column(String(20), nullable=False)
    online_security = Column(String(20))
    online_backup = Column(String(20))
    device_protection = Column(String(20))
    tech_support = Column(String(20))
    streaming_tv = Column(String(20))
    streaming_movies = Column(String(20))
    contract = Column(String(20), nullable=False)
    paperless_billing = Column(String(5), nullable=False)
    payment_method = Column(String(30), nullable=False)
    monthly_charges = Column(Numeric(10, 2), nullable=False)
    total_charges = Column(Numeric(10, 2))

    # Observed outcome (the label)
    churn = Column(Boolean, nullable=False)
    retention_action = Column(Boolean, nullable=False, default=False, index=True)
    label_source = Column(String(20), nullable=False)  # historical | observed
    observed_at = Column(Date, nullable=False)
    snapshot_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Monitoring only (predicted vs actual), never used as features
    prediction_id = Column(Integer)
    predicted_probability = Column(Numeric(5, 4))
    model_version = Column(String(50))

    created_at = Column(DateTime, default=datetime.utcnow)
