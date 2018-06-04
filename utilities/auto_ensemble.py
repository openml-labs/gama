"""
TODO: use scikit-learn components to create the ensemble.
created multiple ensembles from randomly selected subset of models, and average assembles.
"""

from collections import namedtuple
import os
import pickle
import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..ea.evaluation import string_to_metric, evaluate


log = logging.getLogger(__name__)
Model = namedtuple("Model", ['name', 'pipeline', 'predictions'])


class Ensemble(object):

    def __init__(self, model_library_directory, metric, y_true, start_size=1):
        """

        :param model_library_directory: a directory containing results of model evaluations.
        :param metric: metric to optimize the ensemble towards.
        :param y_true: the true labels for the predictions made by the models in the library.
        :param start_size: the top `start_size` models will be included at the start of ensemble building.
        """
        if isinstance(metric, str):
            metric = string_to_metric(metric)

        self._model_library_directory = model_library_directory
        self._metric = metric
        self._y_true = y_true
        self._start_size = start_size
        self._models = None
        self._fit_models = None

    def build(self, n_models_in_ensemble):
        """ Constructs an ensemble out of a library of models. """
        if self._start_size > n_models_in_ensemble:
            raise ValueError("`n_models_in_ensemble` cannot be smaller than start_size specified."
                             "Values were respectively {} and {}".format(n_models_in_ensemble, self._start_size))

        models = load_predictions(self._model_library_directory)
        self._models = build_ensemble(models, self._metric, self._y_true,
                                      start_size=self._start_size, end_size=n_models_in_ensemble)

    def fit(self, X, y):
        """ Constructs an Ensemble out of the library of models.

        :param X: Data to fit the final selection of models on.
        :param y: Targets corresponding to features X.
        :return: None.
        """
        if self._models is None:
            raise RuntimeError("You need to call `build` to select models for the ensemble, before fitting them.")

        self._fit_models = [model.pipeline.fit(X, y) for model in self._models]

        return self

    def predict(self, X):
        predictions = np.argmax(self.predict_proba(X), axis=1)
        return np.squeeze(predictions)

    def predict_proba(self, X):
        predictions = []
        for model in self._fit_models:
            if hasattr(model, 'predict_proba'):
                predictions.append(model.predict_proba(X))
            else:
                class_prediction = model.predict(X)
                ohe_prediction = OneHotEncoder().fit_transform(class_prediction.reshape(-1, 1)).todense()
                predictions.append(np.array(ohe_prediction))

        if len(self._fit_models) == 1:
            return predictions[0]
        else:
            all_predictions = np.stack(predictions)
            return np.mean(all_predictions, axis=0)


def load_predictions(cache_dir):
    models = []
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(cache_dir, file), 'rb') as fh:
                pl, predictions, score = pickle.load(fh)
                predictions = np.array(predictions)
            models.append(Model(str(pl), pl, predictions))
    return models


def evaluate_ensemble(ensemble, metric, y_true):
    """ Evaluates the ensemble according to the metric.

    Currently assumes a single prediction value (e.g. numeric response or positive class probability).
    """
    all_predictions = np.stack([model.predictions for model in ensemble])
    average_predictions = np.mean(all_predictions, axis=0)
    return evaluate(metric, y_true, average_predictions)


def build_ensemble(models, metric, y_true, start_size=0, end_size=5, maximize=True):
    """

    :param models: list of models
    :param metric: metric (y_true, y_pred)
    :param y_true:
    :param start_size: use top n models as start of ensemble
    :param end_size: desired total size of the ensemble
    :param maximize: True if metric should be maximized, False otherwise.
    :return:
    """
    if start_size > end_size:
        raise ValueError('Size must be at least match n. Size: {}, n: {}.'.format(end_size, start_size))

    sorted_ensembles = sorted(models, key=lambda m: evaluate(metric, y_true, m.predictions))
    sorted_ensembles = reversed(sorted_ensembles) if maximize else sorted_ensembles
    ensemble = list(sorted_ensembles)[:start_size]
    best_ensemble = ensemble

    while len(ensemble) < end_size:
        best_ensemble_score = -float('inf') if maximize else float('inf')
        for model in models:
            candidate_ensemble = ensemble + [model]
            candidate_ensemble_score = evaluate_ensemble(candidate_ensemble, metric, y_true)
            if ((maximize and best_ensemble_score < candidate_ensemble_score) or
                    (not maximize and best_ensemble_score > candidate_ensemble_score)):
                best_ensemble, best_ensemble_score = candidate_ensemble, candidate_ensemble_score
        ensemble = best_ensemble
        log.debug('Ensemble size {} , best score: {}'.format(len(ensemble), best_ensemble_score))

    return best_ensemble
