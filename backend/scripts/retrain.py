"""Monthly retraining, following the protocol of the study (docs/paper/articulo.tex):

- Train on the ORIGINAL training split; any resampling lives inside an imblearn
  pipeline, so it is applied only to the training folds of each CV iteration.
- Candidates are compared by cross-validated F1-macro (never by the test set).
- The active model is re-scored on the same folds, so the promotion gate is a
  like-for-like comparison.
- The test split is used only to report metrics of the model that is kept.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

# Add parent to path for app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from app.config import get_settings

settings = get_settings()

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml_pipeline", "data", "resampled", "original"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ml_pipeline", "data", "results")
CV_FOLDS = 5
SEED = 42

# Final model selected in the study: Logistic Regression, no resampling, C=1, L1.
FINAL_MODEL_NAME = "LogisticRegression"


def final_model():
    return LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=1000, random_state=SEED)


# name -> (estimator, sampler or None, param grid). Each family uses the resampling
# strategy that worked best for it in the study.
CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=SEED),
        None,
        {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        RandomOverSampler(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [10, 20]},
    ),
    "XGBoost": (
        xgboost.XGBClassifier(random_state=SEED, eval_metric="logloss"),
        SMOTE(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=SEED),
        SMOTEENN(random_state=SEED),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1]},
    ),
}


def load_data():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))["Churn"]
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))["Churn"]
    return X_train, y_train, X_test, y_test


def cv_splitter():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def with_background(model, X_train: pd.DataFrame):
    """Attach the training mean as SHAP background (used by LinearExplainer at serving time)."""
    model.shap_background_ = X_train.mean().to_frame().T
    return model


def test_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Plain floats: numpy scalars can't be bound as SQL parameters
    return {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }


def register_in_mlflow(model, name: str, params: dict, metrics: dict, run_name: str):
    """Log + register the model and point the production alias to it. Returns the run id."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    version = mlflow.register_model(f"runs:/{run_id}/model", settings.REGISTRY_MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(
        settings.REGISTRY_MODEL_NAME, settings.MODEL_ALIAS, version.version
    )
    print(f"MLflow Registry: {settings.REGISTRY_MODEL_NAME} v{version.version} -> @{settings.MODEL_ALIAS}")
    return run_id


def activate_in_db(db, name: str, version: str, params: dict, metrics: dict, resampling: str | None):
    for m in db.query(ModelRegistry).filter(ModelRegistry.is_active == True).all():
        m.is_active = False
    db.add(ModelRegistry(
        model_name=name,
        model_version=version,
        f1_score=metrics["f1_macro"],
        recall=metrics["recall"],
        precision_score=metrics["precision"],
        auc_roc=metrics["auc_roc"],
        resampling_strategy=resampling or "none",
        hyperparameters=params,
        threshold=settings.MODEL_THRESHOLD,
        is_active=True,
        activated_at=datetime.utcnow(),
    ))


def active_model_cv_f1(X_train, y_train):
    """CV F1-macro of the model currently in production, refit on the same folds."""
    model = mlflow.sklearn.load_model(f"models:/{settings.REGISTRY_MODEL_NAME}@{settings.MODEL_ALIAS}")
    scores = cross_val_score(clone(model), X_train, y_train, cv=cv_splitter(), scoring="f1_macro")
    return float(scores.mean())


def run_retrain():
    db = SessionLocal()
    run_id = str(uuid.uuid4())[:8]
    history = RetrainHistory(run_id=run_id, started_at=datetime.utcnow(), status="running")
    db.add(history)
    db.commit()

    try:
        X_train, y_train, X_test, y_test = load_data()
        history.total_customers = len(X_train) + len(X_test)
        history.train_samples = len(X_train)
        history.test_samples = len(X_test)

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # 1. Select the best candidate by cross-validated F1-macro
        best = None
        for name, (estimator, sampler, grid) in CANDIDATES.items():
            print(f"Training {name}...")
            if sampler is None:
                model, param_grid = estimator, grid
            else:
                model = Pipeline([("sampler", sampler), ("clf", estimator)])
                param_grid = {f"clf__{k}": v for k, v in grid.items()}

            search = GridSearchCV(model, param_grid, cv=cv_splitter(), scoring="f1_macro", n_jobs=-1)
            search.fit(X_train, y_train)
            print(f"  {name}: CV F1-macro={search.best_score_:.4f} {search.best_params_}")

            try:
                with mlflow.start_run(run_name=f"retrain_{name}_{run_id}"):
                    mlflow.log_params(search.best_params_)
                    mlflow.log_metric("cv_f1_macro", search.best_score_)
            except Exception as e:
                print(f"  MLflow logging failed: {e}")

            if best is None or search.best_score_ > best["cv_f1"]:
                best = {
                    "name": name,
                    "cv_f1": float(search.best_score_),
                    "params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
                    "sampler": type(sampler).__name__ if sampler else None,
                    "search": search,
                }

        history.models_evaluated = len(CANDIDATES)
        history.best_model_name = best["name"]

        # Serve the bare classifier: samplers only act at fit time, and SHAP's
        # tree/linear explainers need the estimator itself, not the pipeline.
        fitted = best["search"].best_estimator_
        model = fitted.named_steps["clf"] if isinstance(fitted, Pipeline) else fitted
        model = with_background(model, X_train)
        metrics = test_metrics(model, X_test, y_test)
        history.best_f1_score = metrics["f1_macro"]
        history.best_recall = metrics["recall"]
        history.best_auc_roc = metrics["auc_roc"]
        print(f"Best: {best['name']} (CV F1={best['cv_f1']:.4f}, test F1={metrics['f1_macro']:.4f})")

        # 2. Promotion gate: compare CV F1 against the active model on the same folds
        improved = True
        try:
            current = active_model_cv_f1(X_train, y_train)
            improved = best["cv_f1"] > current + settings.RETRAIN_MIN_IMPROVEMENT
            print(f"Active model CV F1={current:.4f} -> {'promote' if improved else 'keep current'}")
        except Exception as e:
            # No model registered yet: the new one becomes the first production model
            print(f"Could not score the active model ({e}); promoting the new one.")
        history.model_improved = improved

        if improved:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            joblib.dump(model, os.path.join(RESULTS_DIR, f"best_model_{best['name']}.joblib"))

            mlflow_run_id = None
            try:
                mlflow_run_id = register_in_mlflow(
                    model, best["name"], best["params"],
                    {**metrics, "cv_f1_macro": best["cv_f1"]},
                    run_name=f"register_{best['name']}_{run_id}",
                )
            except Exception as e:
                print(f"MLflow registration failed (model still active in app DB): {e}")

            activate_in_db(db, best["name"], mlflow_run_id or run_id, best["params"], metrics, best["sampler"])
            history.best_model_mlflow_run_id = mlflow_run_id or run_id
            print(f"New model activated: {best['name']}")
        else:
            print("Keeping current model.")

        history.status = "success"
        history.finished_at = datetime.utcnow()
        db.commit()
        print(f"\nRetrain completed: {run_id}")
        return history

    except Exception as e:
        history.status = "failed"
        history.error_message = str(e)
        history.finished_at = datetime.utcnow()
        db.commit()
        print(f"Retrain failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_retrain()
