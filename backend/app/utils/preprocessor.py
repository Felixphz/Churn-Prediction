import pandas as pd

# Must match the exact order from model_training.ipynb
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERICAL_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Exact feature names after pd.get_dummies(drop_first=True)
# Derived from the notebook output (line ~306 of data_preprocessing.ipynb)
EXPECTED_FEATURE_NAMES = [
    "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
    "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes",
    "MultipleLines_No phone service", "MultipleLines_Yes",
    "InternetService_Fiber optic", "InternetService_No",
    "OnlineSecurity_No internet service", "OnlineSecurity_Yes",
    "OnlineBackup_No internet service", "OnlineBackup_Yes",
    "DeviceProtection_No internet service", "DeviceProtection_Yes",
    "TechSupport_No internet service", "TechSupport_Yes",
    "StreamingTV_No internet service", "StreamingTV_Yes",
    "StreamingMovies_No internet service", "StreamingMovies_Yes",
    "Contract_One year", "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check",
]


def customer_to_record(customer) -> dict:
    """snake_case row (Customer or TrainingSample) -> notebook CapitalCase record."""
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


def preprocess_customers(records: list[dict]) -> pd.DataFrame:
    """Transform raw customer rows to the 30-feature format expected by the model.

    Equivalent to pd.get_dummies(drop_first=True).astype(int) from data_preprocessing.ipynb,
    but builds each dummy explicitly from EXPECTED_FEATURE_NAMES. Calling get_dummies on a
    single row would drop its only category and leave every categorical feature at 0.
    """
    df = pd.DataFrame([{col: r.get(col) for col in FEATURE_COLUMNS} for r in records])

    # Ensure numeric types
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    encoded = {}
    for name in EXPECTED_FEATURE_NAMES:
        if name in NUMERICAL_COLUMNS:
            encoded[name] = df[name]
        else:
            col, value = name.split("_", 1)
            encoded[name] = df[col] == value

    # astype(int) as in the notebook (it also truncates MonthlyCharges/TotalCharges)
    return pd.DataFrame(encoded, columns=EXPECTED_FEATURE_NAMES).astype(int)


def preprocess_customer(customer_data: dict) -> pd.DataFrame:
    """Single-customer version of preprocess_customers."""
    return preprocess_customers([customer_data])
