import numpy as np

from .gama import Gama
from gama.configuration.regression import reg_config
from gama.utilities.auto_ensemble import EnsembleRegressor


class GamaRegressor(Gama):
    def __init__(self, config=None, objectives=None, *args, **kwargs):
        if not config:
            config = reg_config
        if not objectives:
            objectives = ('neg_mean_squared_error', 'size')
        super().__init__(*args, **kwargs, config=config, objectives=objectives)

    def predict(self, X):
        X = self._preprocess_predict_X(X)
        return self.ensemble.predict(X)

    def _initialize_ensemble(self):
        self.ensemble = EnsembleRegressor(self._scoring_function, self.y_train,
                                          model_library_directory=self._cache_dir, n_jobs=self._n_jobs)
