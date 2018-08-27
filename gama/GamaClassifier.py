import inspect
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
from gama.configuration.classification import clf_config
from gama.ea.metrics import Metric
from gama.utilities.auto_ensemble import EnsembleClassifier


class GamaClassifier(Gama):
    def __init__(self, config=None, objectives=None, *args, **kwargs):
        if not config:
            config = clf_config
        if not objectives:
            objectives = ('neg_log_loss', 'size')

        if Metric(objectives[0]).requires_probabilities:
            # we don't want classifiers that do not have `predict_proba`, because then we have to
            # start doing one hot encodings of predictions etc.
            config = {alg: hp for (alg, hp) in config.items()
                      if not (inspect.isclass(alg) and issubclass(alg, ClassifierMixin)and not hasattr(alg(), 'predict_proba'))}

        self._label_encoder = None
        super().__init__(*args, **kwargs, config=config, objectives=objectives)

    def predict(self, X):
        X = self._preprocess_predict_X(X)
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        """ Predict the class probabilities for input X.

        Predict target for X, using the best found pipeline(s) during the `fit` call.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        X = self._preprocess_predict_X(X)
        return self.ensemble.predict_proba(X)

    def fit(self, X, y, warm_start=False, auto_ensemble_n=25, restart=False, keep_cache=False):
        # Allow arbitrary class name (e.g. string or 1-indexed)
        self._label_encoder = LabelEncoder().fit(y)
        y = self._label_encoder.transform(y)

        super().fit(X, y, warm_start, auto_ensemble_n, restart)

    def _initialize_ensemble(self):
        self.ensemble = EnsembleClassifier(self._scoring_function, self.y_train,
                                           model_library_directory=self._cache_dir, n_jobs=self._n_jobs)

    def _build_fit_ensemble(self, ensemble_size, timeout):
        super()._build_fit_ensemble(ensemble_size, timeout)
        self.ensemble._label_encoder = self._label_encoder