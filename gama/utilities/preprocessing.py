import logging
import pandas as pd
from typing import Optional, Iterator, List, Tuple
from sklearn.base import TransformerMixin

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


def basic_pipeline_extension(
    is_classification: bool
) -> List[Tuple[str, TransformerMixin]]:
    """
    TODO: remove altogether
    """
    return []
