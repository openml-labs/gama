""" This module contains functions for loading data. """
from collections import OrderedDict
import csv
from typing import Tuple, Optional, Dict, List

import arff
import pandas as pd

from gama.data_formatting import infer_categoricals_inplace

CSV_SNIFF_SIZE = 2 ** 12


def sniff_csv_meta(file_path: str) -> Tuple[str, bool]:
    """ Determine the csv delimiter and whether it has a header """
    with open(file_path, "r") as csv_file:
        first_bytes = csv_file.read(2 ** 12)
    sep = csv.Sniffer().sniff(first_bytes).delimiter
    has_header = csv.Sniffer().has_header(first_bytes)
    return sep, has_header


def load_csv_header(file_path: str, **kwargs) -> List[str]:
    """ Return column names in the header, or 0...N if no header is present. """
    if not file_path.endswith(".csv"):
        raise ValueError(f"{file_path} is not a file with .csv extension.")
    sep, has_header = sniff_csv_meta(file_path)
    sep = kwargs.get("sep", sep)

    with open(file_path, "r") as csv_file:
        first_line = csv_file.readline()[:-1]

    if has_header:
        return first_line.split(sep)
    else:
        return [str(i) for i, _ in enumerate(first_line.split(sep))]


def csv_to_pandas(file_path: str, **kwargs) -> pd.DataFrame:
    """ Load data from the csv file into a pd.DataFrame.

    Parameters
    ----------
    file_path: str
        Path of the csv file
    kwargs:
        Additional arguments for pandas.read_csv.
        If not specified, the presence of the header and the delimiter token are
        both detected automatically.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the data in the ARFF file,
        with categorical columns having category dtype.
    """
    if "header" not in kwargs or "sep" not in kwargs:
        sep, has_header = sniff_csv_meta(file_path)
        kwargs["sep"] = kwargs.get("sep", sep)
        kwargs["header"] = kwargs.get("header", 0 if has_header else None)

    df = pd.read_csv(file_path, **kwargs).infer_objects()
    # Since CSV files do not have type annotation, we must infer their type to
    # know which preprocessing steps to apply.
    infer_categoricals_inplace(df)
    return df


def arff_to_pandas(
    file_path: str, encoding: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """ Load data from the ARFF file into a pd.DataFrame.

    Parameters
    ----------
    file_path: str
        Path of the ARFF file
    encoding: str, optional
        Encoding of the ARFF file.
    **kwargs:
        Any arugments for arff.load.

    Returns
    -------
    pandas.DataFrame
        A dataframe of the data in the ARFF file,
        with categorical columns having category dtype.
    """
    with open(file_path, "r", encoding=encoding) as arff_file:
        arff_dict = arff.load(arff_file, **kwargs)

    attribute_names, data_types = zip(*arff_dict["attributes"])
    data = pd.DataFrame(arff_dict["data"], columns=attribute_names)
    for attribute_name, dtype in arff_dict["attributes"]:
        # 'real' and 'numeric' are probably interpreted correctly.
        # Date support needs to be added.
        if isinstance(dtype, list):
            data[attribute_name] = data[attribute_name].astype("category")
    return data


def file_to_pandas(
    file_path: str, encoding: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """ Load ARFF/csv file into pd.DataFrame.

    Parameters
    ----------
    file_path: str
        path to the csv or ARFF file.
    encoding: str, optional
        Encoding, only used for ARFF files.
    kwargs:
        Any arguments for arff.load or pandas.read_csv

    Returns
    -------
    pd.DataFrame
    """
    if file_path.endswith(".arff"):
        data = arff_to_pandas(file_path, encoding, **kwargs)
    elif file_path.endswith(".csv"):
        data = csv_to_pandas(file_path, **kwargs)
    else:
        raise ValueError("Only csv and arff files supported.")
    return data


def X_y_from_file(
    file_path: str,
    split_column: Optional[str] = None,
    encoding: Optional[str] = None,
    **kwargs,
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
        Encoding, only used for ARFF files.
    kwargs:
        Any arguments for arff.load or pandas.read_csv

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (everything except split_column) and targets (split_column).
    """
    data = file_to_pandas(file_path, encoding, **kwargs)
    if split_column is None:
        return data.iloc[:, :-1], data.iloc[:, -1]
    elif split_column in data.columns:
        return data.loc[:, data.columns != split_column], data.loc[:, split_column]
    else:
        raise ValueError(f"No column named {split_column} found in {file_path}")


def load_feature_metadata_from_file(file_path: str) -> Dict[str, str]:
    """ Load the header of the csv or ARFF file, return the type of each attribute.

    For csv files, presence of a header is detected with the Python csv parser.
    If no header is present in the csv file, the columns will be labeled with a number.
    Additionally, the column types is not inferred for csv files.
    """
    if file_path.lower().endswith(".arff"):
        return load_feature_metadata_from_arff(file_path)
    elif file_path.lower().endswith(".csv"):
        return {c: "" for c in load_csv_header(file_path)}
    else:
        raise ValueError("Only csv and arff files supported.")


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
