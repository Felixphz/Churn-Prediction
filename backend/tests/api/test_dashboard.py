"""Smoke tests: the dashboard endpoints answer with and without data."""
from tests.conftest import make_customer


def test_endpoints_work_on_an_empty_database(client):
    summary = client.get("/api/v1/dashboard/summary")
    assert summary.status_code == 200
    assert summary.json()["total_customers"] == 0
    assert client.get("/api/v1/dashboard/at-risk").json()["total"] == 0
    assert client.get("/api/v1/dashboard/analytics").status_code == 200


def test_endpoints_reflect_scored_customers(client, db, fake_model):
    for i in range(3):
        make_customer(db, customer_id=f"C-{i}")
        client.post("/api/v1/predictions/predict", json={"customer_id": f"C-{i}", "threshold": 0.0})

    summary = client.get("/api/v1/dashboard/summary").json()
    assert summary["total_customers"] == 3
    assert summary["customers_scored"] == 3
    assert summary["total_predictions"] == 3

    at_risk = client.get("/api/v1/dashboard/at-risk", params={"level": "todos"}).json()
    for row in at_risk["customers"]:
        assert row["revenue_at_risk"] == round(row["churn_probability"] * row["monthly_charges"], 2)

    analytics = client.get("/api/v1/dashboard/analytics").json()
    assert set(analytics) == {"customers", "predictions", "model"}


def test_invalid_risk_level_returns_422(client):
    assert client.get("/api/v1/dashboard/at-risk", params={"level": "extremo"}).status_code == 422
