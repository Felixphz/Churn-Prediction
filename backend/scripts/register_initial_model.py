"""Bootstrap: register the local best model into the MLflow Model Registry.

Usage (from repo root, with MLFLOW_TRACKING_URI pointing to the target server):
    python backend/scripts/register_initial_model.py
"""
import os
import sys

import joblib
import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings

settings = get_settings()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "ml_pipeline", "data", "results",
    f"best_model_{settings.DEFAULT_MODEL}.joblib",
)


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model file not found: {os.path.abspath(MODEL_PATH)}")

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")
    model = joblib.load(MODEL_PATH)

    with mlflow.start_run(run_name=f"bootstrap_{settings.DEFAULT_MODEL}") as run:
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        mlflow_run_id = run.info.run_id

    model_version = mlflow.register_model(
        f"runs:/{mlflow_run_id}/model", settings.REGISTRY_MODEL_NAME
    )
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        settings.REGISTRY_MODEL_NAME, settings.MODEL_ALIAS, model_version.version
    )
    print(
        f"OK: {settings.REGISTRY_MODEL_NAME} v{model_version.version} "
        f"-> @{settings.MODEL_ALIAS} (run {mlflow_run_id})"
    )


if __name__ == "__main__":
    main()
