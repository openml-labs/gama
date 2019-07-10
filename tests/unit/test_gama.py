import itertools
import numpy as np
import pandas as pd
import pytest

import gama

@pytest.fixture
def gamaclassifier():
    gc = gama.GamaClassifier()
    gc.delete_cache()
    yield gc


def test_reproducible_initialization():
    g1 = gama.GamaClassifier(random_state=1, keep_analysis_log=False)
    pop1 = [g1._operator_set.individual() for _ in range(10)]

    g2 = gama.GamaClassifier(random_state=1, keep_analysis_log=False)
    pop2 = [g2._operator_set.individual() for _ in range(10)]
    for ind1, ind2 in zip(pop1, pop2):
        assert ind1.pipeline_str() == ind2.pipeline_str(), "The initial population should be reproducible."


def test_format_x_y(gamaclassifier):
    """ Tests that X and y data correctly get converted to (pd.DataFrame, pd.Series). """
    def well_formatted_x_y(x, y):
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(x) == len(y)
        assert y.dtype == np.dtype('int64')

    from sklearn.datasets import load_digits
    X_np, y_np = load_digits(return_X_y=True)
    X_df, y_df = pd.DataFrame(X_np), pd.DataFrame(y_np)
    y_str = np.asarray([str(yi) for yi in y_np])
    y_series = pd.Series(y_np)

    for X, y in itertools.product([X_np, X_df], [y_np, y_series, y_df, y_str]):
        well_formatted_x_y(*gamaclassifier._format_x_y(X, y))



