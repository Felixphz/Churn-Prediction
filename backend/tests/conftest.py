"""Shared fixtures. Everything runs locally: SQLite for both DBs and MLflow, a small
in-memory model for inference, and mocks for GitHub / the retrainer service.

The environment is set BEFORE importing `app`: get_settings() is cached and the DB
engines are created at import time, so this also keeps tests away from backend/.env.
"""
import os
import sys
import tempfile
import uuid
from datetime import date

_TMP = tempfile.mkdtemp(prefix="churn-tests-")
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP}/operational.db",
    TRAINING_DATABASE_URL=f"sqlite:///{_TMP}/training.db",
    MLFLOW_TRACKING_URI=f"sqlite:///{_TMP}/mlflow.db",
    ENVIRONMENT="test",
    ENABLE_SCHEDULER="false",
    SEED_ON_STARTUP="false",
    GITHUB_TOKEN="",
    GITHUB_REPO="",
    RETRAIN_SERVICE_URL="http://127.0.0.1:9",
    RETRAIN_MIN_SAMPLES="50",
    TOP_K_MODELS="5",
    MLFLOW_DISABLE_AGENT_HINT="1",
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.training_database import TrainingBase, TrainingSessionLocal, training_engine  # noqa: E402
# Register every model with its metadata (same list as alembic/env.py)
from app.models.customer import Customer  # noqa: E402
from app.models.prediction import Prediction  # noqa: E402,F401
from app.models.model_registry import ModelRegistry  # noqa: E402,F401
from app.models.retrain_history import RetrainHistory  # noqa: E402,F401
from app.models.api_log import ApiLog  # noqa: E402,F401
from app.models.training_sample import TrainingSample  # noqa: E402
from app.utils.model_predictor import ModelLoader  # noqa: E402
from app.utils.preprocessor import EXPECTED_FEATURE_NAMES  # noqa: E402

settings = get_settings()


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(engine)
    TrainingBase.metadata.create_all(training_engine)
    yield


@pytest.fixture(autouse=True)
def clean_db():
    yield
    for metadata, eng in ((Base.metadata, engine), (TrainingBase.metadata, training_engine)):
        with eng.begin() as conn:
            for table in reversed(metadata.sorted_tables):
                conn.execute(table.delete())
    ModelLoader.reset_for_testing()


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def training_db():
    session = TrainingSessionLocal()
    yield session
    session.close()


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_no_raise():
    """Returns 500 responses instead of re-raising server errors."""
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def fake_model(monkeypatch):
    """Small LogisticRegression on the 30 features, served as if loaded from MLflow."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.integers(0, 2, size=(200, len(EXPECTED_FEATURE_NAMES))), columns=EXPECTED_FEATURE_NAMES)
    X["tenure"] = rng.integers(0, 72, size=200)
    X["MonthlyCharges"] = rng.integers(18, 120, size=200)
    X["TotalCharges"] = rng.integers(0, 8000, size=200)
    # Noisy labels: fiber and short tenure raise the churn odds without saturating them
    odds = 0.15 + 0.35 * X["InternetService_Fiber optic"] + 0.3 * (X["tenure"] < 12)
    y = (rng.random(200) < odds).astype(int)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    model.shap_background_ = X.mean().to_frame().T
    ModelLoader._apply_model(model, "1")
    monkeypatch.setattr(ModelLoader, "_fetch_alias_version", classmethod(lambda cls: "1"))
    return model


@pytest.fixture
def mlflow_ns(monkeypatch):
    """Isolated MLflow experiment + registered model name per test (shared SQLite store)."""
    import scripts.retrain as rt
    suffix = uuid.uuid4().hex[:8]
    monkeypatch.setattr(rt, "EXPERIMENT_NAME", f"churn_test_{suffix}")
    monkeypatch.setattr(settings, "REGISTRY_MODEL_NAME", f"churn-model-{suffix}")
    return rt


def customer_payload(**overrides) -> dict:
    data = {
        "customer_id": "C-0001",
        "gender": "Female",
        "senior_citizen": False,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 5,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 85.7,
        "total_charges": 428.5,
    }
    data.update(overrides)
    return data


def make_customer(db, **overrides) -> Customer:
    customer = Customer(**customer_payload(**overrides))
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


def make_samples(training_db, n: int, churn_rate: float = 0.3, treated: int = 0, seed: int = 0):
    """n varied labeled samples; the first `treated` ones had a retention action."""
    rng = np.random.default_rng(seed)
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    samples = []
    for i in range(n):
        churn = i < int(n * churn_rate)
        net = internet[i % 3] if not churn else "Fiber optic"
        addon = "No internet service" if net == "No" else ("Yes" if rng.random() > 0.5 else "No")
        tenure = int(rng.integers(0, 12)) if churn else int(rng.integers(12, 72))
        samples.append(TrainingSample(
            customer_id=f"S-{seed}-{i}",
            **{k: v for k, v in customer_payload(
                gender="Male" if i % 2 else "Female",
                senior_citizen=bool(i % 5 == 0),
                tenure=tenure,
                internet_service=net,
                online_security=addon, online_backup=addon, device_protection=addon,
                tech_support=addon, streaming_tv=addon, streaming_movies=addon,
                contract="Month-to-month" if churn else contracts[i % 3],
                monthly_charges=float(rng.integers(20, 110)),
                total_charges=float(rng.integers(20, 5000)),
            ).items() if k != "customer_id"},
            churn=churn,
            retention_action=i < treated,
            label_source="historical",
            observed_at=date(2026, 1, 1),
        ))
    training_db.add_all(samples)
    training_db.commit()
    return samples
