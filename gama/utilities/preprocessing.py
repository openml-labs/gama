import logging
from typing import Optional, Iterator, List, Tuple
import category_encoders as ce
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dirty_cat import SuperVectorizer

log = logging.getLogger(__name__)


def select_categorical_columns(
    df: pd.DataFrame,
    min_f: Optional[int] = None,
    max_f: Optional[int] = None,
    ignore_nan: bool = True,
) -> Iterator[str]:
    """Find all categorical columns with at least `min_f` and at most `max_f` factors.

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


def basic_encoding(
    x: pd.DataFrame, is_classification: bool
) -> Tuple[pd.DataFrame, TransformerMixin]:
    """Perform 'basic' encoding of categorical features.

    Specifically, perform:
     - Ordinal encoding for features with 2 or fewer unique values.
       FIXME: feature of dirty_cat 0.3 (which is not out as of August 2022)
     - One hot encoding for features with at most 10 unique values.
     - Ordinal encoding for features with 11+ unique values, if y is categorical.
    """
    ohe = ce.OneHotEncoder(handle_missing="value")
    ord_enc = ce.OrdinalEncoder(drop_invariant=True)

    sv = SuperVectorizer(
        cardinality_threshold=11,
        low_card_cat_transformer=ohe,
        high_card_cat_transformer=ord_enc if is_classification else ohe,
    )
    x_enc = sv.fit_transform(x)
    return x_enc, sv


def basic_pipeline_extension(
    x: pd.DataFrame, is_classification: bool
) -> List[Tuple[str, TransformerMixin]]:
    """Define a TargetEncoder and SimpleImputer.

    TargetEncoding is will encode categorical features with more than 10 unique values,
    if y is not categorical. SimpleImputer imputes with the median.
    """
    # These steps need to be in the pipeline because they need to be trained each fold.
    extension_steps = []
    if not is_classification:
        # TargetEncoder is broken with categorical target
        many_factor_features = list(select_categorical_columns(x, min_f=11))
        extension_steps.append(
            ("target_enc", ce.TargetEncoder(cols=many_factor_features))
        )
    extension_steps.append(("imputation", SimpleImputer(strategy="median")))

    return extension_steps
