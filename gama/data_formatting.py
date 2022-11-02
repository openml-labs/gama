from typing import Union, Type, Tuple, Optional

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from gama.utilities.preprocessing import log


def series_looks_categorical(series) -> bool:
    if series.dtype == "object":
        return True
    if is_numeric_dtype(series):
        value_counts = series.value_counts()
        integer_like = series.dtype.kind == "i" or all(
            x.is_integer() for x in series.dropna()
        )
        return len(value_counts) <= 10 and integer_like
    return False


def infer_categoricals_inplace(df):
    """ Use simple heuristics to guess which columns should be categorical. """
    for column in df:
        if series_looks_categorical(df[column]):
            df[column] = df[column].astype("category")


def numpy_to_dataframe(x: np.ndarray) -> pd.DataFrame:
    x = pd.DataFrame(x).infer_objects()
    x = x.infer_objects()
    infer_categoricals_inplace(x)
    return x


def format_y(y: Union[pd.DataFrame, pd.Series, np.ndarray] = None, y_type: Optional[pd.Series] = None):
    """ Transforms a target vector or indicator matrix to a single series (or 1d df) """
    if y is not None:
        if not isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
            raise TypeError("y must be np.ndarray, pd.Series or pd.DataFrame.")
        if y_type not in [pd.Series, pd.DataFrame]:
            raise ValueError(f"`y_type` must be pd.Series or pd.DataFrame but is {y_type}.")

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
        pass
    return y


def remove_unlabeled_rows(
    x: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray] = None
) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    """ Removes all rows from x and y where y is nan. """
    if y is not None:
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

    else:
        pass

    return x, y


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

    if isinstance(x, np.ndarray):
        x = numpy_to_dataframe(x)
    if not isinstance(y, y_type):
        y = format_y(y, y_type)

    if remove_unlabeled:
        x, y = remove_unlabeled_rows(x, y)

    return x, y