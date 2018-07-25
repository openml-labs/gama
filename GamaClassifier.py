import numpy as np
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
from .utilities.configuration import clf_config


class GamaClassifier(Gama):
    def __init__(self, config=None, objectives=None, *args, **kwargs):
        if not config:
            config = clf_config
        if not objectives:
            objectives = ('neg_log_loss', 'size')
        self._label_encoder = None
        super().__init__(*args, **kwargs, config=config, objectives=objectives)

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        predictions = np.squeeze(predictions)
        if self._label_encoder:
            return np.asarray(self._label_encoder.inverse_transform(predictions))
        else:
            return predictions

    def fit(self, X, y, warm_start=False, auto_ensemble_n=25, restart=False):
        # Allows y input in list and pandas series form. Multi-label and DataFrame does not work.
        if hasattr(y, 'values') and isinstance(y.values[0], str):
            self._label_encoder = LabelEncoder().fit(y)
            y = self._label_encoder.transform(y)
        super().fit(X, y, warm_start, auto_ensemble_n, restart)
