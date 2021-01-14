import itertools

import numpy as np

from gama.utilities.preprocessing import select_categorical_columns

import pandas as pd
import pytest

from gama.data_formatting import format_x_y, format_y, series_looks_categorical


class TestFormatY:
    def test_valid_conversions(self):
        y_ins = [
            pd.DataFrame([[0], [0], [1], [1]]),
            pd.Series([0, 0, 1, 1]),
            np.asarray([0, 0, 1, 1]),
            np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]]),
        ]
        out_types = [pd.DataFrame, pd.Series]
        for y_in, out_type in itertools.product(y_ins, out_types):
            y_out = format_y(y_in, y_type=out_type)
            assert isinstance(y_out, out_type)
            assert [0, 0, 1, 1] == list(y_out.values)

    def test_format_y_to_series_categorical(self):
        y_in = pd.Series([0, 0, 1, 1])
        y_out = format_y(y_in, y_type=pd.Series)
        assert isinstance(y_out, pd.Series)
        assert pd.api.types.is_integer_dtype(y_out)

    def test_format_y_to_series_reals(self):
        y_in = pd.Series([0, 0, 0.5, 1, 1])
        y_out = format_y(y_in, y_type=pd.Series)
        assert isinstance(y_out, pd.Series)
        assert y_out.dtype == float


class TestFormatXy:
    def test_format_x_y(self):
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
            well_formatted_x_y(
                *format_x_y(X, y, y_type=pd.DataFrame), y_type=pd.DataFrame
            )

    def test_format_x_y_missing_targets(self):
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


class TestSeriesLooksCategorical:
    def test_object_series(self):
        s = pd.Series(["a", "b"], dtype=object)
        assert series_looks_categorical(s)

    def test_float_series(self):
        s = pd.Series([0.5, 0.8])
        assert not series_looks_categorical(s)

    def test_integer_few_unique_values(self):
        s = pd.Series([1, 2, 3] * 5)
        assert series_looks_categorical(s)

    def test_integer_many_unique_values(self):
        s = pd.Series(range(15))
        assert not series_looks_categorical(s)


def test_find_categorical_columns():
    twelve = pd.Series(list(range(1, 13)), dtype="category", name="twelve")
    six = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], dtype="category", name="six")
    two = pd.Series([1, 2] * 6, dtype="category", name="two")
    two_nan = pd.Series([1, 2, np.nan] * 4, dtype="category", name="two_nan")
    df = pd.DataFrame({s.name: s for s in [two, two_nan, six, twelve]})
    assert ["two", "two_nan"] == list(select_categorical_columns(df, max_f=2))
    assert ["two"] == list(select_categorical_columns(df, max_f=2, ignore_nan=False))
    assert ["six"] == list(select_categorical_columns(df, min_f=5, max_f=10))
    assert ["twelve"] == list(select_categorical_columns(df, min_f=10))
