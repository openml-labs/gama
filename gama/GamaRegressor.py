from .gama import Gama
from gama.configuration.regression import reg_config
from gama.utilities.auto_ensemble import EnsembleRegressor


class GamaRegressor(Gama):
    def __init__(self, config=None, scoring='neg_mean_squared_error', *args, **kwargs):
        if not config:
            config = reg_config
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def predict(self, X=None, arff_file_path=None):
        """ Predict the target for input X.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with predictions. The array is of shape (N,) where N is the length of the
            first dimension of X.
        """
        X = self._preprocess_predict_X(X)
        regressor = self.ensemble if self._ensemble_fit else self._best_pipeline
        return regressor.predict(X)

    def _initialize_ensemble(self):
        self.ensemble = EnsembleRegressor(self._metrics[0], self._y,
                                          model_library_directory=self._cache_dir, n_jobs=self._n_jobs)
