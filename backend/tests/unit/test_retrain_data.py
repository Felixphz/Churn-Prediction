"""Training data comes only from confirmed outcomes in the training DB."""
import numpy as np
import pytest

import scripts.retrain as rt
from tests.conftest import make_samples


def test_load_data_requires_minimum_samples(training_db):
    make_samples(training_db, 20)
    with pytest.raises(ValueError, match="RETRAIN_MIN_SAMPLES"):
        rt.load_data()


def test_treated_customers_are_excluded_by_default(training_db, monkeypatch):
    make_samples(training_db, 120, treated=20)
    X_train, X_test, _, _ = rt.load_data()
    assert len(X_train) + len(X_test) == 100

    monkeypatch.setattr(rt.settings, "RETRAIN_INCLUDE_TREATED", True)
    X_train, X_test, _, _ = rt.load_data()
    assert len(X_train) + len(X_test) == 120


def test_split_is_80_20_and_stratified(training_db):
    make_samples(training_db, 200, churn_rate=0.25)
    X_train, X_test, y_train, y_test = rt.load_data()
    assert (len(X_train), len(X_test)) == (160, 40)
    assert list(X_train.columns) == list(X_test.columns)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_metrics_are_plain_floats(training_db, monkeypatch):
    # Regression (325c34c): numpy scalars can't be bound as SQL parameters. Which sklearn
    # metrics return numpy types depends on the version (roc_auc_score on 1.6.1), so force
    # all of them to numpy and check test_metrics converts every one.
    for name in ("f1_score", "recall_score", "precision_score", "roc_auc_score"):
        original = getattr(rt, name)
        monkeypatch.setattr(rt, name, lambda *a, _f=original, **k: np.float64(_f(*a, **k)))
    make_samples(training_db, 100)
    X_train, X_test, y_train, y_test = rt.load_data()
    model = rt.final_model().fit(X_train, y_train)
    metrics = rt.test_metrics(model, X_test, y_test)
    assert set(metrics) == {"f1_macro", "recall", "precision", "auc_roc"}
    assert all(type(v) is float for v in metrics.values())
    assert not any(isinstance(v, np.generic) for v in metrics.values())
