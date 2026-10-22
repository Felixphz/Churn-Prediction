"""Bootstrap: train the final model of the study and put it in production.

Trains Logistic Regression without resampling (C=1, L1) on the original training
split, registers it in the MLflow Model Registry under the production alias and
marks it active in the app DB.

Usage (from repo root, with MLFLOW_TRACKING_URI pointing to the target server):
    python backend/scripts/register_initial_model.py
"""
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain import (
    FINAL_MODEL_NAME, RESULTS_DIR, activate_in_db, final_model, load_data,
    register_in_mlflow, test_metrics, with_background,
)


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = with_background(final_model().fit(X_train, y_train), X_train)
    metrics = test_metrics(model, X_test, y_test)
    params = {"C": model.C, "penalty": model.penalty, "solver": model.solver}
    print(f"{FINAL_MODEL_NAME}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(RESULTS_DIR, f"best_model_{FINAL_MODEL_NAME}.joblib"))

    run_id = register_in_mlflow(model, FINAL_MODEL_NAME, params, metrics,
                                run_name=f"bootstrap_{FINAL_MODEL_NAME}")

    try:
        from app.database import SessionLocal
        db = SessionLocal()
        activate_in_db(db, FINAL_MODEL_NAME, run_id, params, metrics, resampling=None)
        db.commit()
        db.close()
        print("App DB: model marked as active")
    except Exception as e:
        print(f"App DB not updated ({e}); the API still serves the model from MLflow.")


if __name__ == "__main__":
    main()
