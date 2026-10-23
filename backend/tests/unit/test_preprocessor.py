"""Contract with data_preprocessing.ipynb: 30 dummy features, CapitalCase records."""
from decimal import Decimal

from app.models.customer import Customer
from app.models.training_sample import TrainingSample
from app.utils.preprocessor import (
    EXPECTED_FEATURE_NAMES, customer_to_record, preprocess_customer, preprocess_customers,
)
from tests.conftest import customer_payload


def record(**overrides):
    return customer_to_record(Customer(**customer_payload(**overrides)))


def test_output_has_exactly_the_30_notebook_features_in_order():
    X = preprocess_customers([record(), record(contract="Two year")])
    assert list(X.columns) == EXPECTED_FEATURE_NAMES
    assert len(EXPECTED_FEATURE_NAMES) == 30
    assert X.shape == (2, 30)


def test_single_row_keeps_its_categories():
    # get_dummies on one row would drop its only category: every dummy must still be built
    X = preprocess_customer(record(
        internet_service="Fiber optic", contract="Two year", payment_method="Electronic check",
        gender="Male", partner="No",
    )).iloc[0]
    assert X["InternetService_Fiber optic"] == 1
    assert X["InternetService_No"] == 0
    assert X["Contract_Two year"] == 1
    assert X["Contract_One year"] == 0
    assert X["PaymentMethod_Electronic check"] == 1
    assert X["PaymentMethod_Mailed check"] == 0
    assert X["gender_Male"] == 1
    assert X["Partner_Yes"] == 0


def test_reference_categories_encode_as_all_zeros():
    # drop_first=True: DSL, Month-to-month and Bank transfer are the dropped baselines
    X = preprocess_customer(record(
        internet_service="DSL", contract="Month-to-month", payment_method="Bank transfer (automatic)",
    )).iloc[0]
    assert X["InternetService_Fiber optic"] == X["InternetService_No"] == 0
    assert X["Contract_One year"] == X["Contract_Two year"] == 0
    assert X[[c for c in EXPECTED_FEATURE_NAMES if c.startswith("PaymentMethod_")]].sum() == 0


def test_numeric_columns_truncate_like_the_notebook():
    rec = record(monthly_charges=85.99, total_charges=None, tenure=7, senior_citizen=True)
    X = preprocess_customer(rec).iloc[0]
    assert X["MonthlyCharges"] == 85  # astype(int) truncates
    assert X["TotalCharges"] == 0  # missing total -> 0
    assert X["tenure"] == 7
    assert X["SeniorCitizen"] == 1


def test_blank_total_charges_becomes_zero():
    rec = record()
    rec["TotalCharges"] = " "
    assert preprocess_customer(rec).iloc[0]["TotalCharges"] == 0


def test_customer_to_record_maps_snake_case_and_decimals():
    rec = customer_to_record(Customer(**customer_payload(
        monthly_charges=Decimal("70.35"), total_charges=Decimal("1397.48"),
    )))
    assert rec["MonthlyCharges"] == 70.35 and isinstance(rec["MonthlyCharges"], float)
    assert rec["TotalCharges"] == 1397.48
    assert rec["InternetService"] == "Fiber optic"
    assert rec["PaymentMethod"] == "Electronic check"
    assert "internet_service" not in rec


def test_customer_to_record_works_for_training_samples():
    fields = {k: v for k, v in customer_payload().items() if k != "customer_id"}
    sample = TrainingSample(customer_id="S-1", churn=True, label_source="observed", **fields)
    assert customer_to_record(sample) == customer_to_record(Customer(**customer_payload()))
