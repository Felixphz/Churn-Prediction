import threading
import time

import numpy as np
import shap
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from app.config import get_settings
from app.utils.preprocessor import preprocess_customer, EXPECTED_FEATURE_NAMES

settings = get_settings()

MODEL_URI = f"models:/{settings.REGISTRY_MODEL_NAME}@{settings.MODEL_ALIAS}"

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def _build_explainer(model):
    """Return a SHAP explainer matched to the model type, or None if unsupported."""
    background = pd.DataFrame(
        [np.zeros(len(EXPECTED_FEATURE_NAMES))], columns=EXPECTED_FEATURE_NAMES
    )

    # Tree models: exact, no background needed (path-dependent baseline)
    try:
        return shap.TreeExplainer(model)
    except Exception:
        pass

    # Linear models (LogisticRegression, ...): closed-form on coefficients
    if hasattr(model, "coef_"):
        try:
            return shap.LinearExplainer(model, background)
        except Exception:
            pass

    # Model-agnostic fallback (NaiveBayes, ...)
    if hasattr(model, "predict_proba"):
        try:
            return shap.PermutationExplainer(
                lambda X: model.predict_proba(X)[:, 1], background
            )
        except Exception:
            pass

    return None


class ModelLoader:
    _model = None
    _explainer = None
    _explainer_error = None
    _model_version = None
    _last_alias_check = 0.0
    _lock = threading.RLock()

    @classmethod
    def _fetch_alias_version(cls):
        client = MlflowClient()
        mv = client.get_model_version_by_alias(
            settings.REGISTRY_MODEL_NAME, settings.MODEL_ALIAS
        )
        return str(mv.version)

    @classmethod
    def _apply_model(cls, model, version):
        cls._model = model
        cls._model_version = version
        cls._explainer = None
        cls._explainer_error = None
        cls._last_alias_check = time.time()
        print(f"Model loaded from MLflow Registry: {MODEL_URI} (v{version})")

    @classmethod
    def get_model(cls):
        with cls._lock:
            if cls._model is None:
                try:
                    version = cls._fetch_alias_version()
                    model = mlflow.sklearn.load_model(MODEL_URI)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not load model '{MODEL_URI}' from MLflow at "
                        f"{settings.MLFLOW_TRACKING_URI}. Make sure a model is registered "
                        f"(run backend/scripts/register_initial_model.py). Error: {e}"
                    )
                cls._apply_model(model, version)
                return cls._model

            now = time.time()
            if now - cls._last_alias_check >= settings.MODEL_REFRESH_TTL_SECONDS:
                cls._last_alias_check = now
                try:
                    version = cls._fetch_alias_version()
                    if version != cls._model_version:
                        model = mlflow.sklearn.load_model(MODEL_URI)
                        cls._apply_model(model, version)
                except Exception as e:
                    print(f"Model alias check failed, serving cached model: {e}")
            return cls._model

    @classmethod
    def get_explainer(cls):
        with cls._lock:
            cls.get_model()
            if cls._explainer is None and cls._explainer_error is None:
                try:
                    explainer = _build_explainer(cls._model)
                    if explainer is None:
                        cls._explainer_error = "no SHAP explainer for this model type"
                    else:
                        cls._explainer = explainer
                except Exception as e:
                    cls._explainer_error = str(e)
                    print(f"Explainer build failed: {e}")
            return cls._explainer

    @classmethod
    def reset_for_testing(cls):
        cls._model = None
        cls._explainer = None
        cls._explainer_error = None
        cls._model_version = None
        cls._last_alias_check = 0.0


def _shap_matrix(explainer, X: pd.DataFrame):
    """Return (n_rows, n_features) SHAP array for the churn class and its base value."""
    shap_values = explainer.shap_values(X)
    # shap_values: (n, n_features) array, (n, n_features, 2) array, or [class0, class1] list
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
    sv = sv.reshape(len(X), -1)

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_arr = np.asarray(base_value, dtype=float).ravel()
        base_value = float(base_arr[1]) if base_arr.size > 1 else float(base_arr[0])
    else:
        base_value = float(base_value)
    return sv, base_value


def _shap_payload(sv_row, base_value: float) -> dict:
    """Build SHAP dict, top features and base value for one row."""
    all_shap = {name: round(float(val), 6) for name, val in zip(EXPECTED_FEATURE_NAMES, sv_row)}

    # Top features by absolute impact
    sorted_features = sorted(all_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [
        {
            "feature": name,
            "shap_value": val,
            "direction": "increases churn" if val > 0 else "decreases churn",
        }
        for name, val in sorted_features[:10]
    ]
    return {
        "top_features": top_features,
        "all_shap_values": all_shap,
        "base_value": round(base_value, 6),
        "shap_available": True,
    }


def _no_shap_payload(reason: str) -> dict:
    return {
        "top_features": [],
        "all_shap_values": {},
        "base_value": None,
        "shap_available": False,
        "shap_unavailable_reason": reason,
    }


def predict_frame(X: pd.DataFrame, threshold: float = 0.3) -> list[dict]:
    """Vectorized prediction + SHAP for several preprocessed rows."""
    model = ModelLoader.get_model()
    probs = model.predict_proba(X)[:, 1]

    # SHAP — degrades gracefully so a failing explainer never breaks prediction
    try:
        explainer = ModelLoader.get_explainer()
        if explainer is None:
            raise RuntimeError(
                ModelLoader._explainer_error or "SHAP explainer unavailable"
            )
        sv, base_value = _shap_matrix(explainer, X)
        explanations = [_shap_payload(row, base_value) for row in sv]
    except Exception as e:
        print(f"SHAP unavailable, returning prediction without explanation: {e}")
        explanations = [_no_shap_payload(str(e))] * len(X)

    results = []
    for prob, expl in zip(probs, explanations):
        churn_prob = float(prob)
        results.append({
            "churn_probability": round(churn_prob, 4),
            "churn_predicted": bool(churn_prob >= threshold),
            "threshold": threshold,
            **expl,
        })
    return results


def predict_customer(customer_data: dict, threshold: float = 0.3) -> dict:
    """Run prediction + SHAP for a single customer."""
    return predict_frame(preprocess_customer(customer_data), threshold)[0]
