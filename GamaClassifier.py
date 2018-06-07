import scipy.stats
import numpy as np

from .gama import Gama
from .utilities.configuration import clf_config


class GamaClassifier(Gama):
    def __init__(self, config=None, objectives=None, *args, **kwargs):
        if not config:
            config = clf_config
        if not objectives:
            objectives = ('neg_log_loss', 'size')
        super().__init__(*args, **kwargs, config=config, objectives=objectives)

    def merge_predictions(self, Y):
        """ Computes predictions from a matrix of predictions, with predictions from a pipeline in each columns. """
        return scipy.stats.mode(Y, axis=1)[0]

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        return np.squeeze(predictions)