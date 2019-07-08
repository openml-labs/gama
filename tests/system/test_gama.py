import pytest

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from gama import GamaClassifier


@pytest.fixture
def gamaclassifier():
    gc = GamaClassifier(random_state=0, max_total_time=120)
    yield gc
    gc.delete_cache()


def _gama_on_digits(gama):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    # Add checks on individuals (reproducibility)
    gama.fit(X_train, y_train)

    # Add checks
    gama.predict(X_test)


def test_full_system_single_core(gamaclassifier):
    gamaclassifier._n_jobs = 1
    _gama_on_digits(gamaclassifier)


def test_full_system_multi_core(gamaclassifier):
    gamaclassifier._n_jobs = 2
    _gama_on_digits(gamaclassifier)
