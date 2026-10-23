from app.models.prediction import Prediction
from tests.conftest import customer_payload, make_customer


def predict_data_payload(**overrides):
    data = {k: v for k, v in customer_payload().items() if k != "customer_id"}
    data.update(threshold=0.3, **overrides)
    return data


def test_predict_returns_probability_and_shap_and_is_saved(client, db, fake_model):
    make_customer(db)
    res = client.post("/api/v1/predictions/predict", json={"customer_id": "C-0001"})
    assert res.status_code == 200
    body = res.json()
    prob = body["prediction"]["churn_probability"]
    assert 0.0 <= prob <= 1.0
    assert body["prediction"]["churn_predicted"] == (prob >= 0.3)
    assert body["explainability"]["shap_available"] is True
    assert len(body["explainability"]["top_features"]) == 10
    assert len(body["explainability"]["all_shap_values"]) == 30
    assert body["prediction_type"] == "on_demand"
    assert db.query(Prediction).count() == 1


def test_threshold_decides_the_label(client, db, fake_model):
    # Low-risk profile so the probability is not saturated at 0 or 1
    make_customer(db, contract="Two year", tenure=60, internet_service="DSL")
    probs = set()
    for threshold in (0.0, 0.3, 0.5, 0.9, 1.0):
        body = client.post("/api/v1/predictions/predict",
                           json={"customer_id": "C-0001", "threshold": threshold}).json()["prediction"]
        assert body["threshold"] == threshold
        assert body["churn_predicted"] == (body["churn_probability"] >= threshold)
        probs.add(body["churn_probability"])
    assert len(probs) == 1  # the threshold changes the label, never the probability
    assert 0.0 < probs.pop() < 1.0


def test_predict_data_is_not_saved(client, db, fake_model):
    res = client.post("/api/v1/predictions/predict-data", json=predict_data_payload())
    assert res.status_code == 200
    assert res.json()["prediction_type"] == "simulation"
    assert db.query(Prediction).count() == 0


def test_predict_data_validates_categories(client, fake_model):
    res = client.post("/api/v1/predictions/predict-data", json=predict_data_payload(contract="Weekly"))
    assert res.status_code == 422


def test_latest_and_history(client, db, fake_model):
    make_customer(db)
    for _ in range(2):
        client.post("/api/v1/predictions/predict", json={"customer_id": "C-0001"})
    latest = client.get("/api/v1/predictions/C-0001")
    assert latest.status_code == 200
    assert 0.0 <= latest.json()["prediction"]["churn_probability"] <= 1.0
    assert client.get("/api/v1/predictions/C-0001/history").json()["total"] == 2


def test_unknown_customer_returns_404(client, fake_model):
    assert client.post("/api/v1/predictions/predict", json={"customer_id": "NOPE"}).status_code == 404
    assert client.get("/api/v1/predictions/NOPE").status_code == 404


def test_missing_model_returns_500_with_clear_message(client_no_raise, db, monkeypatch):
    from app.utils.model_predictor import ModelLoader

    def fail(cls):
        raise RuntimeError("no alias")
    monkeypatch.setattr(ModelLoader, "_fetch_alias_version", classmethod(fail))
    make_customer(db)
    res = client_no_raise.post("/api/v1/predictions/predict", json={"customer_id": "C-0001"})
    assert res.status_code == 500
