"""Monthly retraining, following the protocol of the study (docs/paper/articulo.tex):

- Train on the ORIGINAL training split; any resampling lives inside an imblearn
  pipeline, so it is applied only to the training folds of each CV iteration.
- Candidates are compared by cross-validated F1-macro (never by the test set).
- The active model is re-scored on the same folds, so the promotion gate is a
  like-for-like comparison.
- The test split is used only to report metrics of the model that is kept.
- Data comes from the training DB (confirmed outcomes only, never predictions);
  customers that received a retention action are excluded unless
  RETRAIN_INCLUDE_TREATED, since their outcome is confounded by the intervention.
- Every candidate is logged to MLflow with its model; afterwards only the
  TOP_K_MODELS best runs by CV F1-macro (plus the production one) are kept.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

# Add parent to path for app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from app.models.training_sample import TrainingSample
from app.training_database import TrainingSessionLocal
from app.utils.preprocessor import customer_to_record, preprocess_customers
from app.config import get_settings

settings = get_settings()

CV_FOLDS = 5
SEED = 42
EXPERIMENT_NAME = "churn_prediction"

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
    """Labeled samples from the training DB, split like data_preprocessing.ipynb."""
    db = TrainingSessionLocal()
    try:
        query = db.query(TrainingSample)
        if not settings.RETRAIN_INCLUDE_TREATED:
            query = query.filter(TrainingSample.retention_action == False)
        samples = query.order_by(TrainingSample.id).all()
    finally:
        db.close()
    if len(samples) < settings.RETRAIN_MIN_SAMPLES:
        raise ValueError(
            f"Only {len(samples)} labeled samples in the training DB "
            f"(RETRAIN_MIN_SAMPLES={settings.RETRAIN_MIN_SAMPLES})"
        )

    X = preprocess_customers([customer_to_record(s) for s in samples])
    y = pd.Series([int(s.churn) for s in samples], name="Churn")
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


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


def use_experiment():
    """Activate EXPERIMENT_NAME, restoring it if it was deleted from the UI.

    MLflow refuses to log into a deleted experiment, which used to make every
    candidate run fail silently.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        print(f"MLflow: restored deleted experiment '{EXPERIMENT_NAME}' (id {experiment.experiment_id})")
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_model_run(model, params: dict, metrics: dict, run_name: str, tags: dict | None = None) -> str:
    """Log params, metrics and the serving model as one MLflow run. Returns the run id."""
    use_experiment()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        return run.info.run_id


def register_in_mlflow(run_id: str):
    """Register the run's model and point the production alias to it."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    version = mlflow.register_model(f"runs:/{run_id}/model", settings.REGISTRY_MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(
        settings.REGISTRY_MODEL_NAME, settings.MODEL_ALIAS, version.version
    )
    print(f"MLflow Registry: {settings.REGISTRY_MODEL_NAME} v{version.version} -> @{settings.MODEL_ALIAS}")


def production_run_id(client) -> str | None:
    try:
        return client.get_model_version_by_alias(settings.REGISTRY_MODEL_NAME, settings.MODEL_ALIAS).run_id
    except Exception:
        return None


def ranked_runs(client):
    """Finished runs of the experiment that carry a CV F1-macro, best first."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    return client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' and metrics.cv_f1_macro > 0",
        order_by=["metrics.cv_f1_macro DESC"],
        max_results=1000,
    )


def top_models(keep: int | None = None) -> list[dict]:
    """The best `keep` runs by CV F1-macro, as plain dicts for the API."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    prod = production_run_id(client)
    return [
        {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name,
            "model_name": r.data.tags.get("model_name"),
            "sampler": r.data.tags.get("sampler"),
            "params": r.data.params,
            "metrics": r.data.metrics,
            "is_production": r.info.run_id == prod,
            "started_at": datetime.utcfromtimestamp(r.info.start_time / 1000),
        }
        for r in ranked_runs(client)[: keep or settings.TOP_K_MODELS]
    ]


def prune_runs(keep: int | None = None) -> int:
    """Delete every ranked run outside the top `keep`, except the production one."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    prod = production_run_id(client)
    deleted = 0
    for r in ranked_runs(client)[keep or settings.TOP_K_MODELS:]:
        if r.info.run_id != prod:
            client.delete_run(r.info.run_id)
            deleted += 1
    print(f"MLflow: pruned {deleted} runs, kept top {keep or settings.TOP_K_MODELS} (+ production)")
    return deleted


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
        X_train, X_test, y_train, y_test = load_data()
        history.total_customers = len(X_train) + len(X_test)
        history.train_samples = len(X_train)
        history.test_samples = len(X_test)

        # Fail before the grid search if MLflow is unreachable: nothing could be saved
        use_experiment()

        # 1. Select the best candidate by cross-validated F1-macro
        best = None
        any_logged = False
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

            # Serve the bare classifier: samplers only act at fit time, and SHAP's
            # tree/linear explainers need the estimator itself, not the pipeline.
            fitted = search.best_estimator_
            fitted = fitted.named_steps["clf"] if isinstance(fitted, Pipeline) else fitted
            fitted = with_background(fitted, X_train)
            metrics = test_metrics(fitted, X_test, y_test)
            params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
            sampler_name = type(sampler).__name__ if sampler else None

            mlflow_run_id = None
            try:
                mlflow_run_id = log_model_run(
                    fitted, params,
                    {"cv_f1_macro": float(search.best_score_), **{f"test_{k}": v for k, v in metrics.items()}},
                    run_name=f"retrain_{name}_{run_id}",
                    tags={"model_name": name, "sampler": sampler_name or "none", "retrain_run_id": run_id},
                )
                any_logged = True
            except Exception as e:
                print(f"  MLflow logging failed: {e}")

            if best is None or search.best_score_ > best["cv_f1"]:
                best = {
                    "name": name,
                    "cv_f1": float(search.best_score_),
                    "params": params,
                    "sampler": sampler_name,
                    "metrics": metrics,
                    "mlflow_run_id": mlflow_run_id,
                }

        history.models_evaluated = len(CANDIDATES)
        history.best_model_name = best["name"]
        if not any_logged:
            raise RuntimeError("No candidate run could be logged to MLflow")

        metrics = best["metrics"]
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

        mlflow_run_id = best["mlflow_run_id"]
        if improved:
            # The API serves the registry alias: never mark a model active in the DB
            # unless it was actually promoted in MLflow.
            if mlflow_run_id is None:
                raise RuntimeError(f"Best candidate {best['name']} was not logged to MLflow; cannot promote it")
            register_in_mlflow(mlflow_run_id)

            activate_in_db(db, best["name"], mlflow_run_id, best["params"], metrics, best["sampler"])
            history.best_model_mlflow_run_id = mlflow_run_id
            print(f"New model activated: {best['name']}")
        else:
            print("Keeping current model.")

        try:
            prune_runs()
        except Exception as e:
            print(f"MLflow pruning failed: {e}")

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
    try:
        run_retrain()
    except Exception:
        sys.exit(1)  # non-zero so the scheduler (GitHub Actions) marks the run as failed
