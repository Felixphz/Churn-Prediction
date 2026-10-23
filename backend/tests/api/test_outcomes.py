from app.models.customer import Customer
from app.models.training_sample import TrainingSample
from tests.conftest import make_customer, make_samples


def test_churn_outcome_creates_sample_and_deactivates_customer(client, db, training_db):
    make_customer(db)
    res = client.post("/api/v1/outcomes", json={"customer_id": "C-0001", "churned": True})
    assert res.status_code == 201
    body = res.json()
    assert body["label_source"] == "observed" and body["churn"] is True
    assert body["customer_active"] is False

    sample = training_db.query(TrainingSample).one()
    assert sample.contract == "Month-to-month"  # feature snapshot copied
    db.expire_all()
    assert db.query(Customer).one().active is False


def test_duplicate_outcome_returns_409(client, db):
    make_customer(db)
    payload = {"customer_id": "C-0001", "churned": False, "observed_at": "2026-10-01"}
    assert client.post("/api/v1/outcomes", json=payload).status_code == 201
    assert client.post("/api/v1/outcomes", json=payload).status_code == 409


def test_unknown_customer_returns_404(client):
    assert client.post("/api/v1/outcomes", json={"customer_id": "NOPE", "churned": True}).status_code == 404


def test_stayed_customer_remains_active(client, db):
    make_customer(db)
    body = client.post("/api/v1/outcomes", json={"customer_id": "C-0001", "churned": False}).json()
    assert body["customer_active"] is True


def test_outcome_keeps_last_prediction_for_monitoring_only(client, db, training_db, fake_model):
    make_customer(db)
    prob = client.post("/api/v1/predictions/predict", json={"customer_id": "C-0001"}).json()["prediction"]["churn_probability"]
    body = client.post("/api/v1/outcomes", json={"customer_id": "C-0001", "churned": True}).json()
    assert body["prediction_id"] is not None
    assert abs(body["predicted_probability"] - prob) < 1e-4
    # label is the observed outcome, never the prediction
    assert training_db.query(TrainingSample).one().churn is True


def test_summary_excludes_treated_from_trainable(client, db, training_db):
    make_samples(training_db, 10, churn_rate=0.3)
    make_customer(db)
    client.post("/api/v1/outcomes", json={"customer_id": "C-0001", "churned": False, "retention_action": True})

    summary = client.get("/api/v1/outcomes/summary").json()
    assert summary["total_samples"] == 11
    assert summary["by_source"] == {"historical": 10, "observed": 1}
    assert summary["treated"] == 1
    assert summary["treated_retention_rate"] == 100.0
    assert summary["trainable_samples"] == 10
    assert summary["churn_rate"] == round(3 / 11 * 100, 2)
