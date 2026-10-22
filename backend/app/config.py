from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+psycopg2://user:password@localhost:5432/churn_prediction"
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"
    # Labeled outcomes used only for retraining (separate database)
    TRAINING_DATABASE_URL: str = "postgresql+psycopg2://user:password@localhost:5432/churn_training"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    REGISTRY_MODEL_NAME: str = "churn-model"
    MODEL_ALIAS: str = "production"
    MODEL_REFRESH_TTL_SECONDS: int = 30

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"

    # Model
    MODEL_THRESHOLD: float = 0.30
    DEFAULT_MODEL: str = "LogisticRegression"

    # Retrain
    RETRAIN_MIN_IMPROVEMENT: float = 0.01
    TOP_K_MODELS: int = 5  # MLflow runs kept after each retrain (plus production)
    RETRAIN_SERVICE_URL: str = "http://localhost:8001"
    RETRAIN_INCLUDE_TREATED: bool = False  # train on customers that got a retention action
    RETRAIN_MIN_SAMPLES: int = 500
    RETRAIN_STALE_HOURS: int = 3  # a "running" retrain older than this no longer blocks triggers

    # GitHub Actions retrain (production); empty -> proxy to RETRAIN_SERVICE_URL
    GITHUB_TOKEN: str = ""
    GITHUB_REPO: str = ""
    RETRAIN_WORKFLOW: str = "monthly-retrain.yml"

    # Startup flags
    SEED_ON_STARTUP: bool = True
    ENABLE_SCHEDULER: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
