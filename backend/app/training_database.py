"""Training database: confirmed churn outcomes, separate from the operational DB.

Inference never reads it and retraining never reads predictions: labels here are
only real outcomes (historical dataset or POST /api/v1/outcomes).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import get_settings

settings = get_settings()

training_engine = create_engine(settings.TRAINING_DATABASE_URL, pool_pre_ping=True)
TrainingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=training_engine)


class TrainingBase(DeclarativeBase):
    pass


def get_training_db():
    db = TrainingSessionLocal()
    try:
        yield db
    finally:
        db.close()
