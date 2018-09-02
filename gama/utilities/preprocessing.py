import category_encoders as ce
import pandas as pd


def preprocess_folds(df, y, max_categories_for_one_hot=10, folds=5):
    if isinstance(folds, int):
        pass

    preprocessed_folds = []
    for train, test in folds:
        df_train, y_train = df.loc[train, :],  y[train]
        df_train_transformed, transform_function = preprocess(df_train, y_train)
        df_test_transformed = transform_function(df.loc[test, :])
        preprocessed_folds.append(df_train_transformed, df_test_transformed)

    return preprocessed_folds

def preprocess(df, y, max_categories_for_one_hot=10):
    # exclude target column somehow
    one_hot_columns = []
    target_encoding_columns = []
    for unique_values, dtype, column_index in zip(df.apply(pd.Series.nunique), df.dtypes, df.columns):
        if isinstance(dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            if unique_values <= max_categories_for_one_hot:
                one_hot_columns.append(column_index)
            else:
                target_encoding_columns.append(column_index)

    one_hot_encoder = ce.OneHotEncoder(cols=one_hot_columns, impute_missing=False, handle_unknown='ignore')
    one_hot_encoder.fit(df)
    df_ohe = one_hot_encoder.transform(df)

    target_encoder = ce.TargetEncoder(cols=target_encoding_columns)
    target_encoder.fit(df_ohe, y)
    df_ohe_te = target_encoder.transform(df_ohe)

    #def transform_function(X):
    #    X_1 = one_hot_encoder.transform(X)
    #    X_2 = target_encoder.transform(X_1)
    #    # remove any constant features? if we allow for automatic NA columns
    #    return X_2

    transformed_data = transform_function(df)
    return transformed_data, transform_function


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

    return [one_hot_encoder, target_encoder]


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd

    X, y = load_iris(return_X_y=True)
    X[:, 0] = X[:, 0].astype(int)
    X[:, 1] = (X[:, 1] * 10).astype(int)
    X_df = pd.DataFrame(X)
    X_df[0] = X_df[0].astype('category')
    X_df[1] = X_df[1].astype('category')

    X_p = preprocess(X_df, y)
    print(X_p)