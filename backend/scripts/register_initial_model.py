"""Bootstrap: train the final model of the study and put it in production.

Trains Logistic Regression without resampling (C=1, L1) on the original training
split, registers it in the MLflow Model Registry under the production alias and
marks it active in the app DB.

Usage (from repo root, with MLFLOW_TRACKING_URI pointing to the target server):
    python backend/scripts/register_initial_model.py
"""
import os
import sys

from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.retrain import (
    FINAL_MODEL_NAME, activate_in_db, cv_splitter, final_model, load_data,
    log_model_run, register_in_mlflow, test_metrics, with_background,
)


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = with_background(final_model().fit(X_train, y_train), X_train)
    metrics = test_metrics(model, X_test, y_test)
    params = {"C": model.C, "penalty": model.penalty, "solver": model.solver}
    print(f"{FINAL_MODEL_NAME}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # CV F1-macro ranks the run alongside retrain candidates (top-K pruning)
    cv_f1 = float(cross_val_score(final_model(), X_train, y_train, cv=cv_splitter(), scoring="f1_macro").mean())

    run_id = log_model_run(
        model, params,
        {"cv_f1_macro": cv_f1, **{f"test_{k}": v for k, v in metrics.items()}},
        run_name=f"bootstrap_{FINAL_MODEL_NAME}",
        tags={"model_name": FINAL_MODEL_NAME, "sampler": "none"},
    )
    register_in_mlflow(run_id)

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
