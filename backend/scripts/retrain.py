import uuid
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import xgboost

# Add parent to path for app imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from app.config import get_settings

settings = get_settings()

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "NaiveBayes": GaussianNB(),
    "XGBoost": xgboost.XGBClassifier(random_state=42, eval_metric="logloss"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.1, 1], "penalty": ["l2"], "solver": ["liblinear"]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
    "NaiveBayes": {"var_smoothing": [1e-9, 1e-8]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1]},
    "GradientBoosting": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.1]},
}

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "ml_pipeline", "data", "resampled"
)


def run_retrain():
    db = SessionLocal()
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.utcnow()

    history = RetrainHistory(
        run_id=run_id,
        started_at=started_at,
        status="running",
    )
    db.add(history)
    db.commit()

    try:
        # Load data
        X_train = pd.read_csv(os.path.join(DATA_DIR, "smote_enn", "X_train.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "smote_enn", "y_train.csv"))["Churn"]
        X_test = pd.read_csv(os.path.join(DATA_DIR, "original", "X_test.csv"))
        y_test = pd.read_csv(os.path.join(DATA_DIR, "original", "y_test.csv"))["Churn"]

        history.total_customers = len(X_train) + len(X_test)
        history.train_samples = len(X_train)
        history.test_samples = len(X_test)

        # Train models
        best_f1 = 0
        best_model_name = None
        best_model = None
        best_params = None
        models_evaluated = 0

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        except Exception as e:
            print(f"MLflow connection failed, continuing without tracking: {e}")

        for name, model in MODELS.items():
            print(f"Training {name}...")
            param_grid = PARAM_GRIDS[name]

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)

            y_pred = grid_search.predict(X_test)
            y_proba = grid_search.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)

            models_evaluated += 1

            print(f"  {name}: F1={f1:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")

            # Log to MLflow (optional)
            try:
                with mlflow.start_run(run_name=f"retrain_{name}_{run_id}"):
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metrics({"f1_macro": f1, "recall": recall, "auc_roc": auc})
            except Exception as e:
                print(f"  MLflow logging failed: {e}")

            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_recall = recall
                best_auc = auc

        history.models_evaluated = models_evaluated
        history.best_model_name = best_model_name
        history.best_f1_score = best_f1
        history.best_recall = best_recall
        history.best_auc_roc = best_auc

        # Compare with active model
        active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
        improved = True
        if active and active.f1_score:
            min_improvement = settings.RETRAIN_MIN_IMPROVEMENT
            if best_f1 <= float(active.f1_score) + min_improvement:
                improved = False
                print(f"No improvement: new={best_f1:.4f} vs current={float(active.f1_score):.4f}")

        history.model_improved = improved

        if improved:
            # Deactivate current model
            if active:
                active.is_active = False

            # Save model
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..",
                "ml_pipeline", "data", "results",
                f"best_model_{best_model_name}.joblib"
            )
            joblib.dump(best_model, model_path)

            # Register new model
            new_model = ModelRegistry(
                model_name=best_model_name,
                model_version=run_id,
                f1_score=best_f1,
                recall=best_recall,
                auc_roc=best_auc,
                hyperparameters=best_params,
                threshold=0.3,
                is_active=True,
                activated_at=datetime.utcnow(),
            )
            db.add(new_model)
            history.best_model_mlflow_run_id = run_id
            print(f"New model activated: {best_model_name} (F1={best_f1:.4f})")
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
