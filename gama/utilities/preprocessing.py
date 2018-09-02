import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import Imputer


def define_preprocessing_steps(df, max_extra_features_created=None, max_categories_for_one_hot=10):
    if max_extra_features_created:
        # Will determine max_categories_for_one_hot based on how many total new features would be created.
        raise NotImplementedError()

    one_hot_columns = []
    target_encoding_columns = []
    for unique_values, dtype, column_index in zip(df.apply(pd.Series.nunique), df.dtypes, df.columns):
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            if unique_values > max_categories_for_one_hot:
                target_encoding_columns.append(column_index)
            elif unique_values > 2:
                one_hot_columns.append(column_index)
            else:
                pass  # Binary category or constant feature.

    one_hot_encoder = ce.OneHotEncoder(cols=one_hot_columns, impute_missing=False, handle_unknown='ignore')
    target_encoder = ce.TargetEncoder(cols=target_encoding_columns)
    imputer = Imputer(strategy='median')

    return [one_hot_encoder, target_encoder, imputer]
