import logging
from typing import Optional, Iterator
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def find_categorical_columns(
        df: pd.DataFrame,
        min_f: Optional[int] = None,
        max_f: Optional[int] = None,
        ignore_nan: bool = True
) -> Iterator[str]:
    """ Find all categorical columns with at least `min_f` and at most `max_f` factors.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame to design the encoder for.
    min_f: int, optional (default=None)
        The inclusive minimum number of unique values the column should have to be encoded with the given scheme.
    max_f: int, optional (default=None)
        The inclusive maximum number of unique values the column should have to be encoded with the given scheme.
    ignore_nan: bool (default=True)
        If True, don't count NaN as a unique value. If False, count NaN as a unique value (only once).

    Returns
    -------
    An iterator which iterates over the column names that satisfy the criteria.
    """
    for column in df:
        if isinstance(df[column].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            nfactors = df[column].nunique(dropna=ignore_nan)
            if (min_f is None or min_f <= nfactors) and (max_f is None or nfactors <= max_f):
                yield column


def basic_encoding(x: pd.DataFrame):
    """ Performs 'basic' encoding of categorical features. Ordinal if 2 or fewer unique values, OHE if at most 10. """
    binary_features = list(find_categorical_columns(x, max_f=2))
    few_factor_features = list(find_categorical_columns(x, min_f=3, max_f=10))

    encoding_pipeline = Pipeline(
        steps=[
            ('ord-enc', ce.OrdinalEncoder(cols=binary_features, drop_invariant=True)),
            ('oh-enc', ce.OneHotEncoder(cols=few_factor_features, handle_missing='ignore'))
        ]
    )
    x_enc = encoding_pipeline.fit_transform(x, y=None)  # Is this allowed?
    return x_enc, encoding_pipeline


def basic_pipeline_extension(x: pd.DataFrame):
    """ Defines a TargetEncoder for categorical features with more than 10 unique values, and an imputation step. """
    many_factor_features = list(find_categorical_columns(x, min_f=11))
    return [ce.TargetEncoder(cols=many_factor_features), SimpleImputer(strategy='median')]


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
    ordinal_encoding_columns = []
    for unique_values, dtype, column_index in zip(X_df.apply(pd.Series.nunique), X_df.dtypes, X_df.columns):
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            if unique_values > max_categories_for_one_hot:
                target_encoding_columns.append(column_index)
            elif unique_values < 2:
                # Either a constant feature (which gets dropped),
                # or a feature with one unique value and NaNs, where we will encode NaN as a value
                ordinal_encoding_columns.append(column_index)
            elif unique_values == 2:
                if X_df[column_index].isnull().any():
                    # Two unique values and at least one missing value, we apply OHE.
                    one_hot_columns.append(column_index)
                elif X_df[column_index].dtype.categories.dtype == np.dtype('O'):
                    # Even with just two unique values, we need to still map str to numeric for sklearn friendliness.
                    ordinal_encoding_columns.append(column_index)
                else:
                    # 2 unique values, but has a missing value or is numeric already, move on to next column
                    continue
            else:
                one_hot_columns.append(column_index)

    log.debug(f"Detected {sum(isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype) for dtype in X_df.dtypes)} categorical variables, of which:"
              f" - {len(one_hot_columns)} are encoded with OneHotEncoding"
              f" - {len(ordinal_encoding_columns)} are encoded with OrdinalEncoding"
              f" - {len(target_encoding_columns)} are encoded with TargetEncoding.")
    steps = []
    if ordinal_encoding_columns:
        steps.append(ce.OrdinalEncoder(cols=ordinal_encoding_columns, drop_invariant=True, handle_missing='value'))
    if one_hot_columns:
        steps.append(ce.OneHotEncoder(cols=one_hot_columns, handle_missing='return_nan'))
    if target_encoding_columns:
        steps.append(ce.TargetEncoder(cols=target_encoding_columns, handle_missing='value'))

    # We always train an Imputer so we can impute missing data in the test set even if training data
    # has no missing values. It would be better to only do this for the final pipeline.
    steps.append(SimpleImputer(strategy='median'))
    return steps
