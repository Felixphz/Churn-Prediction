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


def preprocess_customer(customer_data: dict) -> pd.DataFrame:
    """Transform raw customer data to the 30-feature format expected by the model.

    Reproduces the columns of pd.get_dummies(drop_first=True) from data_preprocessing.ipynb.
    The dummies are built explicitly from EXPECTED_FEATURE_NAMES: calling get_dummies on a
    single row would drop its only category (drop_first) and zero out every categorical.
    """
    row = {
        "SeniorCitizen": int(customer_data.get("SeniorCitizen") or 0),
        "tenure": int(customer_data.get("tenure") or 0),
        "MonthlyCharges": float(customer_data.get("MonthlyCharges") or 0.0),
        "TotalCharges": float(customer_data.get("TotalCharges") or 0.0),
    }
    for feature in EXPECTED_FEATURE_NAMES:
        if feature in row:
            continue
        col, category = feature.split("_", 1)
        row[feature] = int(str(customer_data.get(col)) == category)

    return pd.DataFrame([row], columns=EXPECTED_FEATURE_NAMES)
