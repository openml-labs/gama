import logging
from typing import Optional, Iterator, List, Tuple
import category_encoders as ce
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def select_categorical_columns(
    df: pd.DataFrame,
    min_f: Optional[int] = None,
    max_f: Optional[int] = None,
    ignore_nan: bool = True,
) -> Iterator[str]:
    """ Find all categorical columns with at least `min_f` and at most `max_f` factors.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame to design the encoder for.
    min_f: int, optional (default=None)
        The inclusive minimum number of unique values the column should have.
    max_f: int, optional (default=None)
        The inclusive maximum number of unique values the column should have.
    ignore_nan: bool (default=True)
        If True, don't count NaN as a unique value.
        If False, count NaN as a unique value (only once).

    Returns
    -------
    An iterator which iterates over the column names that satisfy the criteria.
    """
    for column in df:
        if isinstance(df[column].dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            nfactors = df[column].nunique(dropna=ignore_nan)
            if (min_f is None or min_f <= nfactors) and (
                max_f is None or nfactors <= max_f
            ):
                yield column


def basic_encoding(x: pd.DataFrame):
    """ Perform 'basic' encoding of categorical features.

     Specifically, perform:
      - Ordinal encoding for features with 2 or fewer unique values.
      - One hot encoding for features with at most 10 unique values.
     """
    binary_features = list(select_categorical_columns(x, max_f=2))
    leq_10_features = list(select_categorical_columns(x, min_f=3, max_f=10))

    encoding_pipeline = Pipeline(
        steps=[
            ("ord-enc", ce.OrdinalEncoder(cols=binary_features, drop_invariant=True)),
            ("oh-enc", ce.OneHotEncoder(cols=leq_10_features, handle_missing="ignore")),
        ]
    )
    x_enc = encoding_pipeline.fit_transform(x, y=None)  # Is this allowed?
    return x_enc, encoding_pipeline


def basic_pipeline_extension(x: pd.DataFrame) -> List[Tuple[str, TransformerMixin]]:
    """ Define a TargetEncoder and SimpleImputer.

    TargetEncoding is will encode categorical features with more than 10 unique values.
    SimpleImputer imputes with the median.
    """
    many_factor_features = list(select_categorical_columns(x, min_f=11))
    return [
        ("target_enc", ce.TargetEncoder(cols=many_factor_features)),
        ("imputation", SimpleImputer(strategy="median")),
    ]
