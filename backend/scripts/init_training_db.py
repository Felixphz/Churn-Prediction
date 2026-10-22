"""Create the training DB schema and load the labeled historical dataset (idempotent).

The historical Kaggle rows are loaded only when the table is empty and the raw CSV
exists (it is gitignored: run this once from a machine that has it).

Usage:
    TRAINING_DATABASE_URL=... python backend/scripts/init_training_db.py
"""
import csv
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func

from app.models.training_sample import TrainingSample
from app.training_database import TrainingBase, TrainingSessionLocal, training_engine
from scripts.seed_customers import CSV_PATH, parse_value

# CSV header -> TrainingSample column
CSV_FIELDS = {
    "gender": "gender", "SeniorCitizen": "senior_citizen", "Partner": "partner",
    "Dependents": "dependents", "tenure": "tenure", "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines", "InternetService": "internet_service",
    "OnlineSecurity": "online_security", "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection", "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv", "StreamingMovies": "streaming_movies",
    "Contract": "contract", "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method", "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
}


def init_training_db():
    TrainingBase.metadata.create_all(training_engine)
    db = TrainingSessionLocal()
    try:
        if db.query(func.count(TrainingSample.id)).scalar():
            print("Training DB already has samples, skipping historical load")
            return
        if not os.path.exists(CSV_PATH):
            print(f"Historical CSV not found ({CSV_PATH}), skipping historical load")
            return

        today = date.today()
        with open(CSV_PATH, "r") as f:
            samples = [
                TrainingSample(
                    customer_id=row["customerID"],
                    **{col: parse_value(row[key], col) for key, col in CSV_FIELDS.items()},
                    churn=row["Churn"] == "Yes",
                    retention_action=False,
                    label_source="historical",
                    observed_at=today,
                )
                for row in csv.DictReader(f)
                # Same cleaning as eda.ipynb: drop the 11 rows with blank TotalCharges
                if row["TotalCharges"].strip()
            ]
        db.add_all(samples)
        db.commit()
        print(f"Loaded {len(samples)} historical samples into the training DB")
    except Exception as e:
        db.rollback()
        print(f"Error initializing training DB: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_training_db()
