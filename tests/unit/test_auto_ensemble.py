from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from gama.postprocessing.ensemble import fit_and_weight


def test_fit_and_weight():
    x, y = load_iris(return_X_y=True)
    good_estimator = LinearSVC()
    bad_estimator = LinearSVC(
        penalty="l1"
    )  # Not supported with default squared hinge loss solving the dual problem

    _, w = fit_and_weight((good_estimator, x, y, 1))
    assert 1 == w
    _, w = fit_and_weight((bad_estimator, x, y, 1))
    assert 0 == w
