""" Contains full system tests for GamaRegressor """
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from gama.postprocessing import EnsemblePostProcessing
from gama.utilities.generic.stopwatch import Stopwatch
from gama import GamaRegressor

FIT_TIME_MARGIN = 1.1
TOTAL_TIME_S = 60

# While we could derive statistics dynamically,
# we want to know if any changes ever happen, so we save them statically.
boston = dict(name="boston", load=load_boston, test_size=127, base_mse=81.790)


def _test_gama_regressor(gama, X_train, X_test, y_train, y_test, data, metric):
    with Stopwatch() as sw:
        gama.fit(X_train, y_train)

    assert (
        TOTAL_TIME_S * FIT_TIME_MARGIN >= sw.elapsed_time
    ), "fit must stay within 110% of allotted time."

    predictions = gama.predict(X_test)
    assert isinstance(predictions, np.ndarray), "predictions should be numpy arrays."
    assert (data["test_size"],) == predictions.shape, "should predict (N,) shape array."

    # Majority classifier on this split achieves 0.6293706293706294
    mse = mean_squared_error(y_test, predictions)
    print(data["name"], metric, "mse:", mse)
    assert (
        data["base_mse"] >= mse
    ), "predictions should be at least as good as predicting mean."
    gama.cleanup("all")


def _test_dataset_problem(data, metric):
    x, y = data["load"](return_X_y=True)
    split_data = train_test_split(x, y, random_state=0)

    gama = GamaRegressor(
        random_state=0,
        max_total_time=TOTAL_TIME_S,
        scoring=metric,
        n_jobs=1,
        max_eval_time=300,
        post_processing=EnsemblePostProcessing(ensemble_size=5),
        store="nothing",
    )
    _test_gama_regressor(gama, *split_data, data, metric)


def test_regression_mean_squared_error():
    """ GamaRegressor works on all-numeric data. """
    _test_dataset_problem(boston, "neg_mean_squared_error")


def test_missing_value_regression():
    """ GamaRegressor works when missing values are present. """
    data = boston
    metric = "neg_mean_squared_error"
    X, y = data["load"](return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    X_train[1:300:2, 0] = X_train[2:300:5, 1] = float("NaN")
    X_test[1:100:2, 0] = X_test[2:100:5, 1] = float("NaN")

    gama = GamaRegressor(
        random_state=0, max_total_time=TOTAL_TIME_S, scoring=metric, store="nothing",
    )
    _test_gama_regressor(gama, X_train, X_test, y_train, y_test, data, metric)
