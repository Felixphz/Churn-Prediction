import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.models.customer import Customer
from app.services.customer_service import create_bulk_customers
from app.schemas.customer import CustomerCreate

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "ml_pipeline", "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def parse_value(value: str, field: str):
    if field == "senior_citizen":
        return value == "1"
    if field in ("tenure",):
        return int(value)
    if field in ("monthly_charges", "total_charges"):
        if value == "" or value == " ":
            return 0.0
        return float(value)
    return value


def seed_customers():
    db = SessionLocal()
    try:
        customers = []
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                customer = CustomerCreate(
                    customer_id=row["customerID"],
                    gender=row["gender"],
                    senior_citizen=parse_value(row["SeniorCitizen"], "senior_citizen"),
                    partner=row["Partner"],
                    dependents=row["Dependents"],
                    tenure=parse_value(row["tenure"], "tenure"),
                    phone_service=row["PhoneService"],
                    multiple_lines=row["MultipleLines"],
                    internet_service=row["InternetService"],
                    online_security=row["OnlineSecurity"],
                    online_backup=row["OnlineBackup"],
                    device_protection=row["DeviceProtection"],
                    tech_support=row["TechSupport"],
                    streaming_tv=row["StreamingTV"],
                    streaming_movies=row["StreamingMovies"],
                    contract=row["Contract"],
                    paperless_billing=row["PaperlessBilling"],
                    payment_method=row["PaymentMethod"],
                    monthly_charges=parse_value(row["MonthlyCharges"], "monthly_charges"),
                    total_charges=parse_value(row["TotalCharges"], "total_charges"),
                )
                customers.append(customer)

        result = create_bulk_customers(db, customers)
        print(f"Successfully seeded {len(result)} customers")
        return result

    except Exception as e:
        db.rollback()
        print(f"Error seeding customers: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_customers()
