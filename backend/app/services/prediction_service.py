from datetime import datetime
from sqlalchemy.orm import Session
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.model_registry import ModelRegistry
from app.utils.model_predictor import predict_customer as _predict_customer
from app.utils.preprocessor import preprocess_customer


def get_active_model(db: Session) -> ModelRegistry:
    return db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()


def predict_on_demand(db: Session, customer_id: str, threshold: float = 0.3) -> dict:
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        return None

    # Build dict matching the CSV columns
    customer_data = {
        "gender": customer.gender,
        "SeniorCitizen": customer.senior_citizen,
        "partner": customer.partner,
        "Partner": customer.partner,
        "Dependents": customer.dependents,
        "tenure": customer.tenure,
        "PhoneService": customer.phone_service,
        "MultipleLines": customer.multiple_lines,
        "InternetService": customer.internet_service,
        "OnlineSecurity": customer.online_security,
        "OnlineBackup": customer.online_backup,
        "DeviceProtection": customer.device_protection,
        "TechSupport": customer.tech_support,
        "StreamingTV": customer.streaming_tv,
        "StreamingMovies": customer.streaming_movies,
        "Contract": customer.contract,
        "PaperlessBilling": customer.paperless_billing,
        "PaymentMethod": customer.payment_method,
        "MonthlyCharges": float(customer.monthly_charges),
        "TotalCharges": float(customer.total_charges) if customer.total_charges else 0.0,
    }

    result = _predict_customer(customer_data, threshold)

    # Get active model info
    active_model = get_active_model(db)
    model_name = active_model.model_name if active_model else "GradientBoosting"
    model_version = active_model.model_version if active_model else None

    # Save prediction
    db_prediction = Prediction(
        customer_id=customer.id,
        model_name=model_name,
        model_version=model_version,
        threshold=threshold,
        churn_probability=result["churn_probability"],
        churn_prediction=result["churn_predicted"],
        shap_values=result["all_shap_values"],
        shap_base_value=result["base_value"],
        top_features=result["top_features"],
        prediction_type="on_demand",
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return {
        "customer_id": customer_id,
        "prediction": {
            "churn_probability": result["churn_probability"],
            "churn_predicted": result["churn_predicted"],
            "threshold": threshold,
            "model": model_name,
            "model_version": model_version,
        },
        "explainability": {
            "top_features": result["top_features"],
            "base_value": result["base_value"],
            "all_shap_values": result["all_shap_values"],
        },
        "predicted_at": db_prediction.predicted_at.isoformat(),
        "prediction_type": "on_demand",
    }


def get_prediction_history(db: Session, customer_id: str) -> list:
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        return [], 0

    predictions = (
        db.query(Prediction)
        .filter(Prediction.customer_id == customer.id)
        .order_by(Prediction.predicted_at.desc())
        .all()
    )
    return predictions, len(predictions)


def get_latest_prediction(db: Session, customer_id: str):
    customer = db.query(Customer).filter(Customer.customer_id == customer_id).first()
    if not customer:
        return None

    return (
        db.query(Prediction)
        .filter(Prediction.customer_id == customer.id)
        .order_by(Prediction.predicted_at.desc())
        .first()
    )
