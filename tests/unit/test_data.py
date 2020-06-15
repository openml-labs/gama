import numpy as np
import pandas as pd
from gama.data import arff_to_pandas, X_y_from_file

NUMERIC_TYPES = [np.int, np.int32, np.int64, np.float]


def test_arff_to_pandas():
    # https://www.openml.org/d/23380
    dataframe = arff_to_pandas("tests/data/openml_d_23380.arff")
    assert isinstance(dataframe, pd.DataFrame)
    assert (2796, 35) == dataframe.shape
    assert 68100 == dataframe.isnull().sum().sum()
    assert 32 == sum([dtype in NUMERIC_TYPES for dtype in dataframe.dtypes])
    assert 3 == sum([dtype.name == "category" for dtype in dataframe.dtypes])


def _test_x_y_d23380(x, y):
    """ Test if types are as expected from https://www.openml.org/d/23380 """
    assert isinstance(x, pd.DataFrame)
    assert (2796, 34) == x.shape
    assert 68100 == x.isnull().sum().sum()
    assert 32 == sum([dtype in NUMERIC_TYPES for dtype in x.dtypes])
    assert 2 == sum([dtype.name == "category" for dtype in x.dtypes])

    assert isinstance(y, pd.Series)
    assert (2796,) == y.shape
    assert 0 == y.isnull().sum()
    assert 6 == len(y.dtype.categories)


def test_X_y_from_csv():
    x, y = X_y_from_file("tests/data/openml_d_23380.csv", split_column="TR")
    _test_x_y_d23380(x, y)


def test_X_y_from_arff():
    x, y = X_y_from_file("tests/data/openml_d_23380.arff", split_column="TR")
    _test_x_y_d23380(x, y)
