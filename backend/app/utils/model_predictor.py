import os
import joblib
import numpy as np
import shap
import pandas as pd
from app.utils.preprocessor import preprocess_customer, EXPECTED_FEATURE_NAMES

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..",
    "ml_pipeline", "data", "results", "best_model_GradientBoosting.joblib"
)


class ModelLoader:
    _model = None
    _explainer = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = joblib.load(MODEL_PATH)
        return cls._model

    @classmethod
    def get_explainer(cls):
        if cls._explainer is None:
            model = cls.get_model()
            # Use a background sample for TreeExplainer
            background = shap.sample(
                pd.DataFrame(np.zeros((1, len(EXPECTED_FEATURE_NAMES))), columns=EXPECTED_FEATURE_NAMES),
                1,
            )
            cls._explainer = shap.TreeExplainer(model, background)
        return cls._explainer


def predict_customer(customer_data: dict, threshold: float = 0.3) -> dict:
    """Run prediction + SHAP for a single customer."""
    model = ModelLoader.get_model()
    explainer = ModelLoader.get_explainer()

    X = preprocess_customer(customer_data)

    # Predict
    proba = model.predict_proba(X)[0]
    churn_prob = float(proba[1])
    churn_pred = churn_prob >= threshold

    # SHAP
    shap_values = explainer.shap_values(X)
    # shap_values shape: (1, n_features) for binary classification with TreeExplainer
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 (churn)
    else:
        sv = shap_values[0]

    base_value = float(explainer.expected_value)
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

    # Build SHAP dict: feature_name -> value
    all_shap = {name: round(float(val), 6) for name, val in zip(EXPECTED_FEATURE_NAMES, sv)}

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
        "churn_probability": round(churn_prob, 4),
        "churn_predicted": bool(churn_pred),
        "threshold": threshold,
        "top_features": top_features,
        "all_shap_values": all_shap,
        "base_value": round(base_value, 6),
    }
