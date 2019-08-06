from abc import ABC

import pandas as pd

from .gama import Gama
from gama.configuration.regression import reg_config


class GamaRegressor(Gama):
    def __init__(self, config=None, scoring='neg_mean_squared_error', *args, **kwargs):
        if not config:
            config = reg_config
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def _predict(self, x: pd.DataFrame):
        """ Predict the target for input X.

        :param x: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with predictions. The array is of shape (N,) where N is the length of the
            first dimension of X.
        """
        return self.model.predict(x)
