import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from gama import GamaClassifier


def _gama_on_digits(gama):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )

    # Add checks on individuals (reproducibility)
    gama.fit(X_train, y_train)

    # Add checks
    _ = gama.predict(X_test)
    y_proba = gama.predict_proba(X_test)

    assert log_loss(y_test, y_proba) == gama.score(X_test, y_test)
    assert log_loss(y_test, y_proba) == gama.score(X_test, pd.Series(y_test))
    assert log_loss(y_test, y_proba) == gama.score(
        X_test, LabelEncoder().fit_transform(y_test.reshape(-1, 1))
    )


def test_full_system_single_core():
    automl = GamaClassifier(
        random_state=0,
        max_total_time=60,
        max_memory_mb=2_000,
        store="nothing",
        n_jobs=1,
    )
    _gama_on_digits(automl)


def test_full_system_multi_core():
    automl = GamaClassifier(
        random_state=0,
        max_total_time=60,
        max_memory_mb=4_000,
        store="nothing",
        n_jobs=2,
    )
    _gama_on_digits(automl)
