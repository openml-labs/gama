import numpy as np
import pandas as pd
import pytest

from gama.data_loading import (
    arff_to_pandas,
    X_y_from_file,
    load_feature_metadata_from_file,
    load_feature_metadata_from_arff,
    sniff_csv_meta,
    csv_to_pandas,
    load_csv_header,
    file_to_pandas,
)

NUMERIC_TYPES = [np.int, np.int32, np.int64, np.float]

# https://www.openml.org/d/23380
METADATA_23380 = {
    "N": "INTEGER",
    "TR": "{EL_500_20g/L,EL_500_4g/L,PP_333_20g/L,PP_333_4g/L,control,methanol_control}",
    "TREE": "{D10,D13,D14,D16,D18,D19,D20,D21,D22,G10,G2,G20,G21,G24,G27,G28,G29,G4,G5,G6,G7,G8,G9,J1,J10,J12,J13,J15,J17,J19,J20,J25,J27,J29,J31,J6,J8,M10,M17,M20,M25,M33,M6,O20,O27,O28,O33,O3O,Q12,Q17,Q19,Q23,Q25,Q3,Q34,Q4,Q5}",
    "BR": "{A,B,C,D,E,F,G,H,I,J}",
    "TL": "REAL",
    "IN": "INTEGER",
    **{f"INTERNODE_{i}": "REAL" for i in range(1, 30)},
}

ARFF_BC = "tests/data/breast_cancer_train.arff"
ARFF_CJS = "tests/data/openml_d_23380.arff"
CSV_CJS_FULL = "tests/data/openml_d_23380.csv"
CSV_CJS = "tests/unit/data/openml_d_23380_500.csv"
CSV_NO_HEADER_CJS = "tests/unit/data/openml_d_23380_500_no_header.csv"
CSV_SEMICOLON_CJS = "tests/unit/data/openml_d_23380_500_semi.csv"


def _test_df_d23380(df):
    assert isinstance(df, pd.DataFrame)
    assert (2796, 35) == df.shape
    assert 68100 == df.isnull().sum().sum()
    assert 32 == sum([dtype in NUMERIC_TYPES for dtype in df.dtypes])
    assert 3 == sum([dtype.name == "category" for dtype in df.dtypes])


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


def _test_df_d23380_500(df):
    """ Checks the properties of the subset of 500 rows of the dataset """ ""
    assert isinstance(df, pd.DataFrame)
    assert (500, 35) == df.shape
    assert 12096 == df.isnull().sum().sum()
    # data types are not checked, as the dataset contains too few rows for an
    # accuracy reading


class TestXyFromFile:
    def test_X_y_from_csv(self):
        x, y = X_y_from_file(CSV_CJS_FULL, split_column="TR")
        _test_x_y_d23380(x, y)

    def test_X_y_from_arff(self):
        x, y = X_y_from_file(ARFF_CJS, split_column="TR")
        _test_x_y_d23380(x, y)

    def test_X_y_from_file_invalid_split_column(self):
        with pytest.raises(ValueError, match="No column named NOT_EXIST found"):
            X_y_from_file(ARFF_CJS, split_column="NOT_EXIST")

    def test_X_y_from_file_default_split_column(self):
        _, y = X_y_from_file(ARFF_CJS)
        assert y.name == "INTERNODE_29"


class TestLoadFeatureMetadata:
    def test_load_feature_metadata_from_arff(self):
        meta = load_feature_metadata_from_arff(ARFF_CJS)
        assert meta == METADATA_23380

    def test_load_feature_metadata_from_arff_whitespace_in_feature_name(self):
        meta = load_feature_metadata_from_arff(ARFF_BC)
        assert "mean radius" in meta

    def test_load_feature_metadata_from_file_arff(self):
        meta = load_feature_metadata_from_file(ARFF_CJS)
        assert meta == METADATA_23380

    def test_load_feature_metadata_from_file_csv(self):
        meta = load_feature_metadata_from_file(CSV_CJS)
        assert list(meta) == list(METADATA_23380)
        assert all(v == "" for v in meta.values())

    def test_load_feature_metadata_from_file_txt(self):
        with pytest.raises(ValueError, match="files supported."):
            load_feature_metadata_from_file("myfile.txt")


class TestLoadCsvHeader:
    def test_load_csv_header(self):
        header = load_csv_header(CSV_CJS)
        assert header == list(METADATA_23380)

    def test_load_csv_header_semicolon_delimiter(self):
        header = load_csv_header(CSV_SEMICOLON_CJS)
        assert header == list(METADATA_23380)

    def test_load_csv_header_no_header(self):
        header = load_csv_header(CSV_NO_HEADER_CJS)
        assert header == [str(i) for i, _ in enumerate(METADATA_23380)]

    def test_load_csv_header_wrong_file_type(self):
        with pytest.raises(ValueError, match=r"\S+ is not a file with .csv extension."):
            load_csv_header(ARFF_CJS)


class TestFileToPandas:
    def test_file_to_pandas_csv(self):
        df = file_to_pandas(CSV_CJS_FULL)
        _test_df_d23380(df)

    def test_file_to_pandas_arff(self):
        df = file_to_pandas(ARFF_CJS)
        _test_df_d23380(df)

    def test_file_to_pandas_invalid(self):
        with pytest.raises(ValueError, match="files supported."):
            file_to_pandas("myfile.txt")


class TestArffToPandas:
    def test_arff_to_pandas(self):
        dataframe = arff_to_pandas(ARFF_CJS)
        _test_df_d23380(dataframe)


class TestCsvToPandas:
    def test_csv_to_pandas(self):
        df = csv_to_pandas(CSV_CJS_FULL)
        _test_df_d23380(df)

    def test_csv_to_pandas_semicolon(self):
        df = csv_to_pandas(CSV_SEMICOLON_CJS)
        assert (500, 35) == df.shape

    def test_csv_to_pandas_no_header(self):
        df = csv_to_pandas(CSV_NO_HEADER_CJS)
        assert (500, 35) == df.shape


class TestSniffCsvMeta:
    def test_sniff_csv_meta_with_header(self):
        sep, header = sniff_csv_meta(CSV_CJS)
        assert "," == sep
        assert header

    def test_sniff_csv_meta_with_semicolon(self):
        sep, header = sniff_csv_meta(CSV_SEMICOLON_CJS)
        assert ";" == sep
        assert header

    def test_sniff_csv_meta_no_header(self):
        sep, header = sniff_csv_meta(CSV_NO_HEADER_CJS)
        assert "," == sep
        assert not header
