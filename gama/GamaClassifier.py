import inspect
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
from gama.data import X_y_from_arff
from gama.configuration.classification import clf_config
from gama.utilities.auto_ensemble import EnsembleClassifier


class GamaClassifier(Gama):
    def __init__(self, config=None, scoring='neg_log_loss', *args, **kwargs):
        if not config:
            # Do this to avoid the whole dictionary being included in the documentation.
            config = clf_config

        self._metrics = self._scoring_to_metric(scoring)
        if any(metric.requires_probabilities for metric in self._metrics):
            # we don't want classifiers that do not have `predict_proba`, because then we have to
            # start doing one hot encodings of predictions etc.
            config = {alg: hp for (alg, hp) in config.items()
                      if not (inspect.isclass(alg) and issubclass(alg, ClassifierMixin)and not hasattr(alg(), 'predict_proba'))}

        self._label_encoder = None
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def _predict(self, x: pd.DataFrame):
        """ Predict the target for input X.

        :param x: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with predictions. The array is of shape (N,) where N is the length of the
            first dimension of X.
        """
        classifier = self.ensemble if self._ensemble_fit else self._best_pipeline
        y = classifier.predict(x)
        # Decode the predicted labels - necessary only if ensemble is not used.
        if self._label_encoder is not None and y[0] not in self._label_encoder.classes_:
            y = self._label_encoder.inverse_transform(y)
        return y

    def _predict_proba(self, x: pd.DataFrame):
        """ Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        :param x: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        classifier = self.ensemble if self._ensemble_fit else self._best_pipeline
        return classifier.predict_proba(x)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        """ Predict the class probabilities for input X.

        Predict target for X, using the best found pipeline(s) during the `fit` call.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            for col in self._X.columns:
                X[col] = X[col].astype(self._X[col].dtype)
        return self._predict_proba(X)

    def predict_proba_arff(self, arff_file_path: str):
        """ Predict the class probabilities for input in the arff_file, must have empty target column.

        Predict target for X, using the best found pipeline(s) during the `fit` call.

        :param arff_file_path: str

        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        X, _ = X_y_from_arff(arff_file_path)
        return self._predict_proba(X)

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)

    def _initialize_ensemble(self):
        self.ensemble = EnsembleClassifier(self._metrics[0], self._y,
                                           model_library_directory=self._cache_dir, n_jobs=self._n_jobs)

    def _build_fit_ensemble(self, ensemble_size, timeout):
        super()._build_fit_ensemble(ensemble_size, timeout)
        self.ensemble._label_encoder = self._label_encoder