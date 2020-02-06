import numpy as np

GKEY_TYPE = "https://metadata.datadrivendiscovery.org/types/GroupingKey"
SGKEY_TYPE = "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"


def timeseries_cv(indices, numfolds):
    n = len(indices)
    foldsize = n / (numfolds + 1)
    for i in range(numfolds):
        train = np.asarray(indices[0:int(foldsize * (i+1))], dtype=int)
        test = np.asarray(indices[int(foldsize * (i+1)):int(foldsize * (i+2))], dtype=int)
        yield train, test


def time_series_groups(X):
    grouping_keys = X.metadata.get_columns_with_semantic_type(GKEY_TYPE)
    suggested_grouping_keys = X.metadata.get_columns_with_semantic_type(SGKEY_TYPE)
    if len(grouping_keys) == 0:
        grouping_keys = suggested_grouping_keys
    # No grouping keys.  All the data represents one time series.
    if len(grouping_keys) == 0:
        yield list(X.index)
        return
    assert len(grouping_keys) == 1
    colname = X.columns[grouping_keys[0]]
    for val, df in X.groupby(colname):
        yield list(df.index)


def timeseries_cv_grouped(X, numfolds):
    for group in time_series_groups(X):
        for train, test in timeseries_cv(group, numfolds):
            yield train, test
        break
