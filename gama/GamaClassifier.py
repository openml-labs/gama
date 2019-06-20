import inspect
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
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

    def predict(self, X=None, arff_file_path=None):
        """ Predict the target for input X.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with predictions. The array is of shape (N,) where N is the length of the
            first dimension of X.
        """
        X = self._preprocess_predict_X(X, arff_file_path)
        classifier = self.ensemble if self._ensemble_fit else self._best_pipeline
        return classifier.predict(X)

    def predict_proba(self, X=None, arff_file_path=None):
        """ Predict the class probabilities for input X.

        Predict target for X, using the best found pipeline(s) during the `fit` call.

        :param X: a 2d numpy array with the length of the second dimension is equal to that of X of `fit`.
        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        X = self._preprocess_predict_X(X, arff_file_path)
        classifier = self.ensemble if self._ensemble_fit else self._best_pipeline
        return classifier.predict_proba(X)

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)

    def _initialize_ensemble(self):
        y = self.y_train
        # y may be a DataFrame.  If so, flatten it.
        if y.ndim != 1:
            y = y.squeeze()
        self.ensemble = EnsembleClassifier(self._metrics[0], y,
                                           model_library_directory=self._cache_dir, n_jobs=self._n_jobs)

    def _build_fit_ensemble(self, ensemble_size, timeout):
        super()._build_fit_ensemble(ensemble_size, timeout)
        self.ensemble._label_encoder = self._label_encoder