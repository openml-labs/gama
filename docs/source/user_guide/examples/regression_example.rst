Regression
**********

::

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from gama import GamaRegressor

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    automl = GamaRegressor(max_total_time=180)
    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)

    print('MSE:', mean_squared_error(y_test, predictions))

Should take 3 minutes to run and give the output below (exact performance might differ)::

    MSE: 19.238475470025886

By default, GamaRegressor will optimize towards `mean squared error`.