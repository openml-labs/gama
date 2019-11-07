import logging
from typing import Tuple, Union, Type, Optional
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

log = logging.getLogger(__name__)


def define_preprocessing_steps(X_df: pd.DataFrame,
                               max_extra_features_created: Optional[int] = None,
                               max_categories_for_one_hot: int = 10):
    """ Constructs imputation and categorical preprocessing steps for the dataframe, taking into account data types.

    Parameters
    ----------
    X_df: pandas.DataFrame
        Features of the data (i.e. without target label).
    max_extra_features_created: int, optional (default=None)
        WARNING! Currently not supported, the only valid value is None.

        If set, dynamically decrease max_categories_for_one_hot as needed so as not to
        create a greater amount of features than max_extra_features_created.
    max_categories_for_one_hot: int (default=10)
        The maximum amount of unique category levels to be considered for one hot encoding.

    Returns
    -------
    List
        List of preprocessing step objects.
    """
    if max_extra_features_created:
        # Will determine max_categories_for_one_hot based on how many total new features would be created.
        raise NotImplementedError()

    one_hot_columns = []
    target_encoding_columns = []
    for unique_values, dtype, column_index in zip(X_df.apply(pd.Series.nunique), X_df.dtypes, X_df.columns):
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            if unique_values > max_categories_for_one_hot:
                target_encoding_columns.append(column_index)
            elif unique_values > 1:
                one_hot_columns.append(column_index)
            else:
                pass  # Binary category or constant feature.

    one_hot_encoder = ce.OneHotEncoder(cols=one_hot_columns, handle_unknown='ignore')
    target_encoder = ce.TargetEncoder(cols=target_encoding_columns, handle_unknown='ignore')
    imputer = SimpleImputer(strategy='median')

    return [one_hot_encoder, target_encoder, imputer]


def heuristic_numpy_to_dataframe(X: np.ndarray, max_unique_values_cat: int = 10) -> pd.DataFrame:
    """ Transform a numpy array to a typed pd.DataFrame. """
    X_df = pd.DataFrame(X)
    for column, n_unique in X_df.nunique(dropna=True).items():
        if n_unique <= max_unique_values_cat:
            X_df[column] = X_df[column].astype('category')
    return X_df


def format_x_y(x: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, pd.Series, np.ndarray],
               y_type: Type=pd.Series, remove_unlabeled: bool = True
               ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series]]:
    """ Takes various types of (X,y) data and converts it into a (pd.DataFrame, pd.Series) tuple.

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
        raise ValueError(f"`y_type` must be one of [pandas.Series, pandas.DataFrame] but is {y_type}.")

    if remove_unlabeled:
        unlabeled = y[y.columns[0]].isnull() if isinstance(y, pd.DataFrame) else y.isnull()
        if unlabeled.isnull().any():
            log.info("Target vector has been found to contain NaN-labels, these rows will be ignored.")
            x, y = x.loc[~y.isnull(), :], y[~y.isnull(), :]

    return x, y
