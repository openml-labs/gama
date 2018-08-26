import inspect
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
from gama.configuration.classification import clf_config
from gama.ea.metrics import Metric


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
        if self._best_pipeline is not None:
            return self._best_pipeline.predict(X)
        raise NotImplementedError
        # Have to rethink how we do Ensembles.
        predictions = np.argmax(self.predict_proba(X), axis=1)
        predictions = np.squeeze(predictions)
        if self._label_encoder:
            return np.asarray(self._label_encoder.inverse_transform(predictions))
        else:
            return predictions

    def fit(self, X, y, warm_start=False, auto_ensemble_n=25, restart=False, keep_cache=False):
        # Allow arbitrary class name (e.g. string or 1-indexed)
        self._label_encoder = LabelEncoder().fit(y)
        y = self._label_encoder.transform(y)

        super().fit(X, y, warm_start, auto_ensemble_n, restart)

    def _build_fit_ensemble(self, ensemble_size, timeout):
        super()._build_fit_ensemble(ensemble_size, timeout)
        self.ensemble._label_encoder = self._label_encoder