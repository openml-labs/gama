import itertools
import pandas as pd
from gama.utilities.preprocessing import format_x_y


def test_format_x_y():
    """ Tests that X and y data correctly get converted to (pd.DataFrame, pd.DataFrame). """
    def well_formatted_x_y(x, y, y_type):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, y_type)
        assert len(x) == len(y)

    from sklearn.datasets import load_digits
    X_np, y_np = load_digits(return_X_y=True)
    X_df, y_df = pd.DataFrame(X_np), pd.DataFrame(y_np)
    y_series = pd.Series(y_np)
    y_2d = y_np.reshape(-1, 1)

    for X, y in itertools.product([X_np, X_df], [y_np, y_series, y_df, y_2d]):
        well_formatted_x_y(*format_x_y(X, y), y_type=pd.Series)
        well_formatted_x_y(*format_x_y(X, y, y_type=pd.DataFrame), y_type=pd.DataFrame)
