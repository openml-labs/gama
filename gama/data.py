""" This module contains functions for loading data. """
from typing import Tuple, Optional

import arff
import pandas as pd


def arff_to_pandas(file_path: str) -> pd.DataFrame:
    """ Load data from the ARFF file into a pd.DataFrame.

    Parameters
    ----------
    file_path: str
        Path of the ARFF file

    Returns
    -------
    pandas.DataFrame
        A dataframe of the data in the ARFF file,
        with categorical columns having category dtype.
    """
    if not isinstance(file_path, str):
        raise TypeError(f"`file_path` must be of type `str` but is of type {type(file_path)}")

    with open(file_path, 'r') as arff_file:
        arff_dict = arff.load(arff_file)

    attribute_names, data_types = zip(*arff_dict['attributes'])
    data = pd.DataFrame(arff_dict['data'], columns=attribute_names)
    for attribute_name, dtype in arff_dict['attributes']:
        # 'real' and 'numeric' are probably interpreted correctly, date support needs to be added.
        if isinstance(dtype, list):
            data[attribute_name] = data[attribute_name].astype('category')
    return data


def X_y_from_arff(file_path: str, split_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """ Load data from the ARFF file into pandas DataFrame and specified column to pd.Series. "

    Parameters
    ----------
    file_path: str
        path to the ARFF file.
    split_column: str, optional (default=None)
        Column to split and return separately (e.g. target column).
        Value should either match a column name or None.
        If None is specified, the last column is returned separately.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (everything except split_column) and targets (split_column).
    """
    data = arff_to_pandas(file_path)

    if split_column is None:
        return data.iloc[:, :-1], data.iloc[:, -1]
    elif split_column in data.columns:
        return data.loc[:, data.columns != split_column], data.loc[:, split_column]
    else:
        raise ValueError("No column with name {} found in ARFF file {}".format(split_column, file_path))
