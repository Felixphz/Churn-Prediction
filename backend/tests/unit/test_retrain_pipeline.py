"""run_retrain end to end with a 1x1 grid, plus its failure modes."""
from unittest import mock

import pytest
from sklearn.linear_model import LogisticRegression

import scripts.retrain as rt
from app.models.model_registry import ModelRegistry
from app.models.retrain_history import RetrainHistory
from tests.conftest import make_samples

TINY_CANDIDATES = {
    "LogisticRegression": (
        LogisticRegression(solver="liblinear", max_iter=1000, random_state=rt.SEED),
        None,
        {"C": [1]},
    ),
}


@pytest.fixture
def tiny_grid(monkeypatch, training_db):
    monkeypatch.setattr(rt, "CANDIDATES", TINY_CANDIDATES)
    make_samples(training_db, 150)


def last_history(db):
    return db.query(RetrainHistory).order_by(RetrainHistory.id.desc()).first()


def test_full_retrain_registers_and_activates_the_model(tiny_grid, mlflow_ns, db):
    history = rt.run_retrain()

    assert history.status == "success"
    assert history.models_evaluated == 1
    assert (history.train_samples, history.test_samples) == (120, 30)
    assert history.model_improved is True  # nothing registered yet

    import mlflow
    assert rt.production_run_id(mlflow.MlflowClient()) == history.best_model_mlflow_run_id
    active = db.query(ModelRegistry).filter(ModelRegistry.is_active == True).one()
    assert active.model_name == "LogisticRegression"
    assert active.model_version == history.best_model_mlflow_run_id


def test_second_retrain_on_same_data_keeps_the_current_model(tiny_grid, mlflow_ns, db):
    rt.run_retrain()
    history = rt.run_retrain()
    assert history.status == "success"
    assert history.model_improved is False
    assert db.query(ModelRegistry).count() == 1


def test_fails_before_training_when_mlflow_is_down(tiny_grid, monkeypatch, db):
    monkeypatch.setattr(rt.settings, "MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    with mock.patch.object(rt, "GridSearchCV") as grid:
        with pytest.raises(Exception):
            rt.run_retrain()
        grid.assert_not_called()  # no compute wasted when nothing could be saved
    assert last_history(db).status == "failed"


def test_fails_when_no_candidate_could_be_logged(tiny_grid, mlflow_ns, monkeypatch, db):
    monkeypatch.setattr(rt, "log_model_run", mock.Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="No candidate run could be logged"):
        rt.run_retrain()
    assert last_history(db).status == "failed"
    assert db.query(ModelRegistry).count() == 0  # never active without MLflow


def test_main_exits_non_zero_on_failure(monkeypatch):
    monkeypatch.setattr(rt, "run_retrain", mock.Mock(side_effect=RuntimeError("boom")))
    with pytest.raises(SystemExit) as exc:
        exec(compile("if __name__ == '__main__':\n" + open(rt.__file__).read().split("if __name__ == \"__main__\":\n", 1)[1],
                     rt.__file__, "exec"), {**rt.__dict__, "__name__": "__main__"})
    assert exc.value.code == 1
