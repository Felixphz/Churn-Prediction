"""MLflow side of the retrain: logging, registry alias, top-K ranking and pruning."""
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression


def tiny_model():
    X = np.random.default_rng(0).random((40, 3))
    return LogisticRegression().fit(X, np.array([0, 1] * 20))


def log(rt, cv_f1, name="LR"):
    """Full run with its model (what the retrain does for every candidate)."""
    return rt.log_model_run(
        tiny_model(), {"C": 1}, {"cv_f1_macro": cv_f1, "test_f1_macro": cv_f1 - 0.01},
        run_name=f"run_{cv_f1}", tags={"model_name": name, "sampler": "none"},
    )


def log_metrics_only(rt, cv_f1, name="LR"):
    """Ranking and pruning only look at metrics: skip the model upload to keep tests fast."""
    rt.use_experiment()
    with mlflow.start_run(run_name=f"run_{cv_f1}", tags={"model_name": name}) as run:
        mlflow.log_metric("cv_f1_macro", cv_f1)
        return run.info.run_id


def set_production(rt, run_id):
    """Point the alias at a run without a model artifact (enough for ranking tests)."""
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(rt.settings.REGISTRY_MODEL_NAME)
    except Exception:
        pass
    version = client.create_model_version(rt.settings.REGISTRY_MODEL_NAME, f"runs:/{run_id}/model", run_id=run_id)
    client.set_registered_model_alias(rt.settings.REGISTRY_MODEL_NAME, rt.settings.MODEL_ALIAS, version.version)


def test_log_model_run_stores_params_metrics_and_model(mlflow_ns):
    rt = mlflow_ns
    run_id = log(rt, 0.71)
    run = mlflow.MlflowClient().get_run(run_id)
    assert run.data.params["C"] == "1"
    assert run.data.metrics["cv_f1_macro"] == 0.71
    assert run.data.tags["model_name"] == "LR"
    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert loaded.predict(np.zeros((1, 3))).shape == (1,)


def test_register_moves_the_production_alias(mlflow_ns):
    rt = mlflow_ns
    first, second = log(rt, 0.70), log(rt, 0.72)  # registering needs a real model artifact
    rt.register_in_mlflow(first)
    rt.register_in_mlflow(second)
    client = mlflow.MlflowClient()
    assert rt.production_run_id(client) == second


def test_prune_keeps_top_k_and_never_the_production_run(mlflow_ns):
    rt = mlflow_ns
    run_ids = [log_metrics_only(rt, 0.60 + i / 100) for i in range(8)]  # 0.60 ... 0.67
    set_production(rt, run_ids[0])  # the worst run is in production

    assert rt.prune_runs(keep=5) == 2

    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(rt.EXPERIMENT_NAME)
    alive = {r.info.run_id for r in client.search_runs([experiment.experiment_id])}
    assert alive == set(run_ids[3:]) | {run_ids[0]}


def test_top_models_are_sorted_and_flag_production(mlflow_ns):
    rt = mlflow_ns
    low, high, mid = (log_metrics_only(rt, 0.65, "A"), log_metrics_only(rt, 0.75, "B"),
                      log_metrics_only(rt, 0.70, "C"))
    set_production(rt, mid)

    top = rt.top_models(keep=3)
    assert [t["model_name"] for t in top] == ["B", "C", "A"]
    assert [t["is_production"] for t in top] == [False, True, False]
    assert top[0]["metrics"]["cv_f1_macro"] == 0.75


def test_use_experiment_restores_a_deleted_experiment(mlflow_ns):
    # Regression: a deleted experiment made every candidate run fail silently
    rt = mlflow_ns
    client = mlflow.MlflowClient()
    experiment_id = client.create_experiment(rt.EXPERIMENT_NAME)
    client.delete_experiment(experiment_id)

    run_id = log_metrics_only(rt, 0.7)

    assert client.get_experiment(experiment_id).lifecycle_stage == "active"
    assert client.get_run(run_id).info.experiment_id == experiment_id
