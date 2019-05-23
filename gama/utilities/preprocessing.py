import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


def define_preprocessing_steps(X_df, max_extra_features_created=None, max_categories_for_one_hot=10):
    """ Constructs the preprocessing steps for the dataframe, taking into account data types.

    :param X_df: Features of the data (i.e. without target label).
    :param max_extra_features_created: If set, dynamically decrease max_categories_for_one_hot as needed so as not to
        create a greater amount of features than max_extra_features_created.
    :param max_categories_for_one_hot: The maximum amount of unique category levels to be considered for one hot encoding.
    :return: a list of preprocessing step objects.
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
    imputer = Imputer(strategy='median')

    return [one_hot_encoder, target_encoder, imputer]


def heuristic_numpy_to_dataframe(X: np.ndarray, max_unique_values_cat: int=10) -> pd.DataFrame:
    """ Transform a numpy array to a typed pd.DataFrame. """
    X_df = pd.DataFrame(X)
    for column, n_unique in X_df.nunique(dropna=True).items():
        if n_unique <= max_unique_values_cat:
            X_df[column] = X_df[column].astype('category')
    return X_df
