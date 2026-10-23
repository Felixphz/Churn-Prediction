from datetime import datetime, timedelta
from unittest import mock

import httpx
import pytest

from app.models.prediction import Prediction
from app.models.retrain_history import RetrainHistory
from app.routers import pipeline
from app.utils import batch_predictor
from tests.conftest import make_customer


@pytest.fixture
def batch_run():
    def run(batch_run_id="b1"):
        batch_predictor.BATCH_RUNS[batch_run_id] = {
            "batch_run_id": batch_run_id, "status": "running", "total_customers": 0, "processed": 0,
            "churn_predicted": 0, "churn_rate": 0, "errors": 0, "error_message": None,
            "started_at": None, "finished_at": None, "duration_seconds": None,
        }
        return batch_predictor.run_batch_predictions(batch_run_id)
    yield run
    batch_predictor.BATCH_RUNS.clear()


def test_batch_scores_only_active_customers(db, fake_model, batch_run):
    make_customer(db, customer_id="A")
    make_customer(db, customer_id="B")
    make_customer(db, customer_id="GONE", active=False)

    result = batch_run()
    assert result["status"] == "completed"
    assert (result["total_customers"], result["processed"], result["errors"]) == (2, 2, 0)
    assert db.query(Prediction).count() == 2


def test_batch_status_is_rebuilt_from_db_after_restart(client, db, fake_model, batch_run):
    make_customer(db)
    batch_run("b2")
    batch_predictor.BATCH_RUNS.clear()  # simulate a process restart

    body = client.get("/api/v1/predictions/batch/b2").json()
    assert (body["status"], body["processed"], body["total_customers"]) == ("completed", 1, 1)
    assert client.get("/api/v1/predictions/batch/unknown").status_code == 404


def add_history(db, status="running", age=timedelta(minutes=5), run_id="r1"):
    db.add(RetrainHistory(run_id=run_id, started_at=datetime.utcnow() - age, status=status))
    db.commit()


def test_trigger_is_blocked_while_a_retrain_is_running(client, db):
    add_history(db)
    assert client.post("/api/v1/retrain/trigger").status_code == 409


def test_stale_running_retrain_does_not_block(client, db):
    add_history(db, age=timedelta(hours=10))
    with mock.patch.object(pipeline, "_trigger_retrainer_service") as local:
        assert client.post("/api/v1/retrain/trigger").status_code == 200
    local.assert_called_once()


def test_trigger_dispatches_github_workflow(client, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "GITHUB_TOKEN", "tok")
    monkeypatch.setattr(pipeline.settings, "GITHUB_REPO", "owner/repo")
    with mock.patch.object(pipeline.httpx, "post", return_value=mock.Mock(status_code=204)) as post:
        res = client.post("/api/v1/retrain/trigger")
    assert res.status_code == 200
    assert res.json() == {"message": "Retrain triggered in background"}
    url = post.call_args.args[0]
    assert url == "https://api.github.com/repos/owner/repo/actions/workflows/monthly-retrain.yml/dispatches"
    assert post.call_args.kwargs["json"] == {"ref": "main"}
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer tok"


def test_github_rejection_returns_502(client, monkeypatch):
    monkeypatch.setattr(pipeline.settings, "GITHUB_TOKEN", "tok")
    monkeypatch.setattr(pipeline.settings, "GITHUB_REPO", "owner/repo")
    with mock.patch.object(pipeline.httpx, "post", return_value=mock.Mock(status_code=401, text="Bad credentials")):
        assert client.post("/api/v1/retrain/trigger").status_code == 502


def test_local_trigger_returns_503_when_retrainer_is_down(client):
    with mock.patch.object(pipeline.httpx, "post", side_effect=httpx.ConnectError("refused")):
        assert client.post("/api/v1/retrain/trigger").status_code == 503


def test_retrain_status_and_history(client, db):
    assert client.get("/api/v1/retrain/status").status_code == 404
    for i in range(3):
        add_history(db, status="success", age=timedelta(days=3 - i), run_id=f"r{i}")

    assert client.get("/api/v1/retrain/status").json()["run_id"] == "r2"
    history = client.get("/api/v1/retrain/history", params={"limit": 2}).json()
    assert history["total"] == 2
    assert [r["run_id"] for r in history["runs"]] == ["r2", "r1"]
