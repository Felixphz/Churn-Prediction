from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, Numeric, DateTime, Text
from app.database import Base


class RetrainHistory(Base):
    __tablename__ = "retrain_history"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), unique=True, nullable=False)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)
    status = Column(String(20), nullable=False)

    # Datos de entrenamiento
    total_customers = Column(Integer)
    train_samples = Column(Integer)
    test_samples = Column(Integer)

    # Resultados
    models_evaluated = Column(Integer)
    best_model_name = Column(String(50))
    best_model_mlflow_run_id = Column(String(50))
    best_f1_score = Column(Numeric(6, 4))
    best_recall = Column(Numeric(6, 4))
    best_auc_roc = Column(Numeric(6, 4))
    model_improved = Column(Boolean, default=False)

    # Errores
    error_message = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
