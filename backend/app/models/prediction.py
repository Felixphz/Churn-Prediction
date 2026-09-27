from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Numeric, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), index=True)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(50))
    threshold = Column(Numeric(5, 4), nullable=False)

    # Predicción
    churn_probability = Column(Numeric(5, 4), nullable=False)
    churn_prediction = Column(Boolean, nullable=False)

    # Explicabilidad SHAP
    shap_values = Column(JSON, nullable=False)
    shap_base_value = Column(Numeric(8, 6))
    top_features = Column(JSON)

    # Metadata
    prediction_type = Column(String(20), nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow, index=True)
    batch_run_id = Column(String(50), index=True)

    # Relationships
    customer = relationship("Customer", back_populates="predictions")
