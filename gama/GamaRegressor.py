import numpy as np

from .gama import Gama
from gama.configuration.regression import reg_config


class GamaRegressor(Gama):
    def __init__(self, config=None, objectives=None, *args, **kwargs):
        if not config:
            config = reg_config
        if not objectives:
            objectives = ('neg_mean_squared_error', 'size')
        super().__init__(*args, **kwargs, config=config, objectives=objectives)

    def predict(self, X):
        predictions = self.ensemble.predict_proba(X)
        return np.squeeze(predictions)
