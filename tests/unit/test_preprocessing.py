import itertools

import numpy as np
import pandas as pd

from gama.data import format_x_y
from gama.utilities.preprocessing import (
    find_categorical_columns,
    basic_encoding,
    basic_pipeline_extension,
)


def test_format_x_y():
    """ X and y data get converted to (pd.DataFrame, pd.DataFrame). """

    def well_formatted_x_y(x, y, y_type):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, y_type)
        assert len(x) == len(y)

    from sklearn.datasets import load_digits

    X_np, y_np = load_digits(return_X_y=True)
    X_df, y_df = pd.DataFrame(X_np), pd.DataFrame(y_np)
    y_series = pd.Series(y_np)
    y_2d = y_np.reshape(-1, 1)

    for X, y in itertools.product([X_np, X_df], [y_np, y_series, y_df, y_2d]):
        well_formatted_x_y(*format_x_y(X, y), y_type=pd.Series)
        well_formatted_x_y(*format_x_y(X, y, y_type=pd.DataFrame), y_type=pd.DataFrame)


def test_format_x_y_missing_targets():
    """ Samples with missing labels should be removed from training data. """

    def well_formatted_x_y(x, y, y_type):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, y_type)
        assert len(x) == len(y)

    from sklearn.datasets import load_digits

    x, y = load_digits(return_X_y=True)
    y = y.astype(float)
    y[::2] = np.nan
    x_, y_ = format_x_y(x, y)

    assert (1797,) == y.shape
    assert (898,) == y_.shape
    assert np.array_equal(y[1::2], y_)
    assert np.array_equal(x[1::2, :], x_)
    well_formatted_x_y(x_, y_, y_type=pd.Series)


def test_find_categorical_columns():
    twelve = pd.Series(list(range(1, 13)), dtype="category", name="twelve")
    six = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype="category", name="six")
    two = pd.Series([1, 2] * 6, dtype="category", name="two")
    two_nan = pd.Series([1, 2, np.nan] * 4, dtype="category", name="two_nan")
    df = pd.DataFrame({s.name: s for s in [two, two_nan, six, twelve]})
    assert ["two", "two_nan"] == list(find_categorical_columns(df, max_f=2))
    assert ["two"] == list(find_categorical_columns(df, max_f=2, ignore_nan=False))
    assert ["six"] == list(find_categorical_columns(df, min_f=5, max_f=10))
    assert ["twelve"] == list(find_categorical_columns(df, min_f=10))
