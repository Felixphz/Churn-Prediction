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

    Replicates the exact pd.get_dummies(drop_first=True) from data_preprocessing.ipynb.
    """
    raw = {col: customer_data.get(col) for col in FEATURE_COLUMNS}

    # Ensure numeric types
    raw["SeniorCitizen"] = int(raw["SeniorCitizen"])
    raw["tenure"] = int(raw["tenure"])
    raw["MonthlyCharges"] = float(raw["MonthlyCharges"])
    raw["TotalCharges"] = float(raw["TotalCharges"]) if raw["TotalCharges"] else 0.0

    df = pd.DataFrame([raw])

    # One-Hot Encoding — must match notebook exactly: get_dummies on cat_cols, drop_first=True
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True).astype(int)

    # Align columns to the expected 30 features (fill missing with 0)
    for col in EXPECTED_FEATURE_NAMES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[EXPECTED_FEATURE_NAMES]
    return df_encoded
