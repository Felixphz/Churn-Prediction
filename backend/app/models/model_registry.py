from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, Numeric, DateTime, JSON
from app.database import Base


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(50), nullable=False)
    mlflow_experiment_id = Column(String(50))
    mlflow_run_uri = Column(String(200))

    # Métricas del modelo
    f1_score = Column(Numeric(6, 4))
    recall = Column(Numeric(6, 4))
    auc_roc = Column(Numeric(6, 4))
    precision_score = Column(Numeric(6, 4))

    # Configuración
    resampling_strategy = Column(String(30))
    hyperparameters = Column(JSON)
    threshold = Column(Numeric(5, 4))
    feature_names = Column(JSON)

    # Estado
    is_active = Column(Boolean, default=False, index=True)
    activated_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
