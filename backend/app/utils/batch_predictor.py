import threading
import uuid
from datetime import datetime

from app.database import SessionLocal
from app.models.customer import Customer
from app.models.prediction import Prediction
from app.models.model_registry import ModelRegistry
from app.utils.model_predictor import predict_frame
from app.utils.preprocessor import preprocess_customers

CHUNK_SIZE = 500

# Estado de los batch en curso / terminados en este proceso (batch_run_id -> dict)
BATCH_RUNS: dict[str, dict] = {}
_runs_lock = threading.Lock()


def _customer_data(customer: Customer) -> dict:
    return {
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


def _update(batch_run_id: str, **fields):
    with _runs_lock:
        BATCH_RUNS[batch_run_id].update(fields)


def run_batch_predictions(batch_run_id: str, threshold: float = 0.3) -> dict:
    """Execute batch predictions for all customers, committing chunk by chunk."""
    started_at = datetime.utcnow()
    db = SessionLocal()
    try:
        customer_ids = [cid for (cid,) in db.query(Customer.id).order_by(Customer.id).all()]
        total = len(customer_ids)
        _update(batch_run_id, total_customers=total)

        active_model = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).first()
        model_name = active_model.model_name if active_model else "GradientBoosting"
        model_version = active_model.model_version if active_model else None

        processed = churn_count = errors = 0
        first_error = None

        for i in range(0, total, CHUNK_SIZE):
            ids = customer_ids[i:i + CHUNK_SIZE]
            customers = db.query(Customer).filter(Customer.id.in_(ids)).all()
            try:
                rows, valid = [], []
                for customer in customers:
                    try:
                        rows.append(_customer_data(customer))
                        valid.append(customer)
                    except Exception as e:
                        errors += 1
                        first_error = first_error or f"{customer.customer_id}: {e}"

                if valid:
                    results = predict_frame(preprocess_customers(rows), threshold)
                    for customer, result in zip(valid, results):
                        db.add(Prediction(
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
                        ))
                        churn_count += result["churn_predicted"]
                    db.commit()
                    processed += len(valid)
            except Exception as e:
                db.rollback()
                errors += len(customers)
                first_error = first_error or str(e)
                print(f"Batch {batch_run_id}: chunk {i // CHUNK_SIZE} failed: {e}")

            _update(batch_run_id, processed=processed, churn_predicted=churn_count,
                    errors=errors, error_message=first_error)

        status = "completed" if processed else "failed"
    except Exception as e:
        status = "failed"
        _update(batch_run_id, error_message=str(e))
        print(f"Batch {batch_run_id} failed: {e}")
    finally:
        db.close()

    finished_at = datetime.utcnow()
    run = BATCH_RUNS[batch_run_id]
    total = run["total_customers"]
    _update(
        batch_run_id,
        status=status,
        churn_rate=round(run["churn_predicted"] / run["processed"] * 100, 2) if run["processed"] else 0,
        finished_at=finished_at.isoformat(),
        duration_seconds=round((finished_at - started_at).total_seconds(), 2),
    )
    print(f"Batch {batch_run_id} {status}: {run['processed']}/{total} customers, {run['errors']} errors")
    return BATCH_RUNS[batch_run_id]


def start_batch_predictions(threshold: float = 0.3) -> dict:
    """Launch a batch run in a background thread and return its initial status."""
    with _runs_lock:
        running = next((r for r in BATCH_RUNS.values() if r["status"] == "running"), None)
        if running:
            return running
        batch_run_id = str(uuid.uuid4())[:8]
        BATCH_RUNS[batch_run_id] = {
            "batch_run_id": batch_run_id,
            "status": "running",
            "total_customers": 0,
            "processed": 0,
            "churn_predicted": 0,
            "churn_rate": 0,
            "errors": 0,
            "error_message": None,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "duration_seconds": None,
        }
    threading.Thread(
        target=run_batch_predictions, args=(batch_run_id, threshold), daemon=True
    ).start()
    return BATCH_RUNS[batch_run_id]
