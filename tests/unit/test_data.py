import numpy as np
import pandas as pd
from gama.data import (
    arff_to_pandas,
    X_y_from_file,
    load_feature_metadata_from_file,
    load_feature_metadata_from_arff,
)

NUMERIC_TYPES = [np.int, np.int32, np.int64, np.float]
METADATA_23380 = {
    "N": "INTEGER",
    "TR": "{EL_500_20g/L,EL_500_4g/L,PP_333_20g/L,PP_333_4g/L,control,methanol_control}",
    "TREE": "{D10,D13,D14,D16,D18,D19,D20,D21,D22,G10,G2,G20,G21,G24,G27,G28,G29,G4,G5,G6,G7,G8,G9,J1,J10,J12,J13,J15,J17,J19,J20,J25,J27,J29,J31,J6,J8,M10,M17,M20,M25,M33,M6,O20,O27,O28,O33,O3O,Q12,Q17,Q19,Q23,Q25,Q3,Q34,Q4,Q5}",
    "BR": "{A,B,C,D,E,F,G,H,I,J}",
    "TL": "REAL",
    "IN": "INTEGER",
    **{f"INTERNODE_{i}": "REAL" for i in range(1, 30)},
}


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


def test_load_feature_metadata_from_file_arff():
    meta = load_feature_metadata_from_file("tests/data/openml_d_23380.arff")
    assert meta == METADATA_23380


def test_load_feature_metadata_from_arff():
    meta = load_feature_metadata_from_arff("tests/data/openml_d_23380.arff")
    assert meta == METADATA_23380


def test_load_feature_metadata_from_file_csv():
    meta = load_feature_metadata_from_file("tests/data/openml_d_23380.csv")
    assert list(meta) == list(METADATA_23380)
    assert all(v == "" for v in meta.values())


def test_load_csv_header():
    meta = load_feature_metadata_from_file("tests/data/openml_d_23380.csv")
    assert list(meta) == list(METADATA_23380)
    assert all(v == "" for v in meta.values())
