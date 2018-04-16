import numpy as np

from gama import Gama
from configuration import reg_config


class GamaRegressor(Gama):
    def __init__(self, config=None, scoring=None, *args, **kwargs):
        if not config:
            config = reg_config
        if not scoring:
            scoring = 'neg_mean_squared_error'
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def merge_predictions(self, Y):
        """ Computes predictions from a matrix of predictions, with predictions from a pipeline in each columns. """
        return np.mean(Y, axis=1)
