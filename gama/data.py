""" This module contains functions for loading data. """
from collections import OrderedDict
import csv
from typing import Tuple, Optional, Dict, Union, Type

import arff
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from gama.utilities.preprocessing import log


def csv_to_pandas(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as csv_file:
        has_header = csv.Sniffer().has_header(csv_file.read(2048))
    df = pd.read_csv(file_path, header=0 if has_header else None)

    # Since CSV files do not have type annotation, we must infer their type to
    # know which preprocessing steps to apply. All `str` columns and int-like columns
    # with <=10 unique values are considered categorical.
    for column, n_unique in df.nunique(dropna=True).items():
        if df[column].dtype == "object":
            df[column] = df[column].astype("category")
        elif n_unique <= 10 and is_numeric_dtype(df[column]):
            for x in df[column].dropna().unique():
                if isinstance(x, float) and not x.is_integer():
                    break
            else:
                df[column] = df[column].astype("category")

    return df


def arff_to_pandas(file_path: str, encoding: Optional[str] = None) -> pd.DataFrame:
    """ Load data from the ARFF file into a pd.DataFrame.

    Parameters
    ----------
    file_path: str
        Path of the ARFF file
    encoding: str, optional
        Encoding of the ARFF file.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the data in the ARFF file,
        with categorical columns having category dtype.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"`file_path` must be of type `str` but is {type(file_path)}")

    with open(file_path, "r", encoding=encoding) as arff_file:
        arff_dict = arff.load(arff_file)

    attribute_names, data_types = zip(*arff_dict["attributes"])
    data = pd.DataFrame(arff_dict["data"], columns=attribute_names)
    for attribute_name, dtype in arff_dict["attributes"]:
        # 'real' and 'numeric' are probably interpreted correctly.
        # Date support needs to be added.
        if isinstance(dtype, list):
            data[attribute_name] = data[attribute_name].astype("category")
    return data


def X_y_from_file(
    file_path: str, split_column: Optional[str] = None, encoding: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """ Load ARFF/csv file into pd.DataFrame and specified column to pd.Series.

    Parameters
    ----------
    file_path: str
        path to the ARFF file.
    split_column: str, optional (default=None)
        Column to split and return separately (e.g. target column).
        Value should either match a column name or None.
        If None is specified, the last column is returned separately.
    encoding: str, optional
        Encoding of the ARFF file.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (everything except split_column) and targets (split_column).
    """
    if file_path.endswith(".arff"):
        data = arff_to_pandas(file_path, encoding)
    elif file_path.endswith(".csv"):
        data = csv_to_pandas(file_path)
    else:
        raise ValueError("Only csv and arff files supported.")

    if split_column is None:
        return data.iloc[:, :-1], data.iloc[:, -1]
    elif split_column in data.columns:
        return data.loc[:, data.columns != split_column], data.loc[:, split_column]
    else:
        raise ValueError(f"No column named {split_column} found in {file_path}")


def load_feature_metadata_from_arff(file_path: str) -> Dict[str, str]:
    """ Load the header of the ARFF file and return the type of each attribute. """
    data_header = "@data"
    attribute_indicator = "@attribute"
    attributes = OrderedDict()
    with open(file_path, "r") as fh:
        line = fh.readline()
        while not line.lower().startswith(data_header):
            if line.lower().startswith(attribute_indicator):
                # arff uses a space separator, but allows spaces in
                # feature name (name must be quoted) and feature type (if nominal).
                indicator, name_and_type = line.split(" ", 1)
                if name_and_type.startswith('"'):
                    name, data_type = name_and_type[1:].split('" ', 1)
                    name = name
                else:
                    name, data_type = name_and_type.split(" ", 1)
                attributes[name] = data_type
            line = fh.readline()[:-1]  # remove newline character
    return attributes


def heuristic_numpy_to_dataframe(
    x: np.ndarray, max_unique_values_cat: int = 10
) -> pd.DataFrame:
    """ Transform a numpy array to a typed pd.DataFrame. """
    x_df = pd.DataFrame(x)
    for column, n_unique in x_df.nunique(dropna=True).items():
        if n_unique <= max_unique_values_cat:
            x_df[column] = x_df[column].astype("category")
    return x_df


def format_x_y(
    x: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
    y_type: Type = pd.Series,
    remove_unlabeled: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    """ Take (X,y) data and convert it to a (pd.DataFrame, pd.Series) tuple.

    Parameters
    ----------
    x: pandas.DataFrame or numpy.ndarray
    y: pandas.DataFrame, pandas.Series or numpy.ndarray
    y_type: Type (default=pandas.Series)
    remove_unlabeled: bool (default=True)
        If true, remove all rows associated with unlabeled data (NaN in y).

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame or pandas.Series]
        X and y, where X is formatted as pd.DataFrame and y is formatted as `y_type`.
    """
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError("X must be either np.ndarray or pd.DataFrame.")
    if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
        raise TypeError("y must be np.ndarray, pd.Series or pd.DataFrame.")

    if isinstance(x, np.ndarray):
        x = heuristic_numpy_to_dataframe(x)
    if isinstance(y, np.ndarray) and y.ndim == 2:
        # Either indicator matrix or should be a vector.
        if y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        else:
            y = y.squeeze()

    if y_type == pd.Series:
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)
    elif y_type == pd.DataFrame:
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
    else:
        raise ValueError(f"`y_type` must be pd.Series or pd.DataFrame but is {y_type}.")

    if remove_unlabeled:
        if isinstance(y, pd.DataFrame):
            unlabeled = y.iloc[:, 0].isnull()
        else:
            unlabeled = y.isnull()

        if unlabeled.any():
            log.info(
                f"Target vector has been found to contain {sum(unlabeled)} NaN-labels, "
                f"these rows will be ignored."
            )
            x, y = x.loc[~unlabeled], y.loc[~unlabeled]

    return x, y
