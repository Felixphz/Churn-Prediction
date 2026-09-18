import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.model_registry import ModelRegistry
from app.utils.model_predictor import predict_customer


def run_batch_predictions(db: Session, threshold: float = 0.3) -> dict:
    """Execute batch predictions for all customers."""
    batch_run_id = str(uuid.uuid4())[:8]
    started_at = datetime.utcnow()

    customers = db.query(Customer).all()
    total = len(customers)
    churn_count = 0
    errors = 0

    active_model = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
    model_name = active_model.model_name if active_model else "GradientBoosting"
    model_version = active_model.model_version if active_model else None

    for customer in customers:
        try:
            customer_data = {
                "gender": customer.gender,
                "SeniorCitizen": customer.senior_citizen,
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

            result = predict_customer(customer_data, threshold)

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
                prediction_type="batch",
                batch_run_id=batch_run_id,
            )
            db.add(db_prediction)

            if result["churn_predicted"]:
                churn_count += 1

        except Exception as e:
            errors += 1
            print(f"Error predicting customer {customer.customer_id}: {e}")

    db.commit()

    finished_at = datetime.utcnow()
    churn_rate = round(churn_count / total * 100, 2) if total > 0 else 0

    return {
        "batch_run_id": batch_run_id,
        "total_customers": total,
        "churn_predicted": churn_count,
        "churn_rate": churn_rate,
        "errors": errors,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
    }
