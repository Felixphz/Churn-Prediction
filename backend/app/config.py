from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/churn_prediction"
    DB_USER: str = "user"
    DB_PASSWORD: str = "password"

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
    DEFAULT_MODEL: str = "GradientBoosting"

    # Retrain
    RETRAIN_MIN_IMPROVEMENT: float = 0.01

    # Startup flags
    SEED_ON_STARTUP: bool = True
    ENABLE_SCHEDULER: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
