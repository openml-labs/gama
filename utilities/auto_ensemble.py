"""
TODO: use scikit-learn components to create the ensemble.
created multiple ensembles from randomly selected subset of models, and average assembles.
"""

from collections import namedtuple
import os
import pickle

import numpy as np

from ..ea.evaluation import string_to_metric

Model = namedtuple("Model", ['name', 'pipeline', 'predictions'])


def load_predictions(cache_dir):
    models = []
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(cache_dir, file), 'rb') as fh:
                pl, predictions, score = pickle.load(fh)
            models.append(Model(str(pl), pl, predictions))
    return models


def evaluate_ensemble(ensemble, metric, y_true):
    """ Evaluates the ensemble according to the metric.

    Currently assumes a single prediction value (e.g. numeric response or positive class probability).
    """
    all_predictions = np.c_[[model.predictions for model in ensemble]].T
    average_predictions = np.mean(all_predictions, axis=1)
    return metric(y_true, average_predictions)


def build_ensemble(models, metric, y_true, n=0, size=5, maximize=True):
    """

    :param models: list of models
    :param metric: metric (y_true, y_pred)
    :param y_true:
    :param n: use top n models as start of ensemble
    :param size: desired total size of the ensemble
    :param maximize: True if metric should be maximized, False otherwise.
    :return:
    """
    if n > size:
        raise ValueError('Size must be at least match n.')

    sorted_ensembles = sorted(models, key=lambda m: metric(y_true, m.predictions))
    sorted_ensembles = reversed(sorted_ensembles) if maximize else sorted_ensembles
    ensemble = list(sorted_ensembles)[:n]
    best_ensemble = ensemble

    while len(ensemble) < size:
        best_ensemble_score = -float('inf') if maximize else float('inf')
        for model in models:
            candidate_ensemble = ensemble + [model]
            candidate_ensemble_score = evaluate_ensemble(candidate_ensemble, metric, y_true)
            if ((maximize and best_ensemble_score < candidate_ensemble_score) or
                    (not maximize and best_ensemble_score > candidate_ensemble_score)):
                best_ensemble, best_ensemble_score = candidate_ensemble, candidate_ensemble_score
        ensemble = best_ensemble
        print('Ensemble size', len(ensemble), ', best score:', best_ensemble_score)

    return best_ensemble


def ensemble_predict(ensemble, X, Xt, yt):
    # TODO: Use scikit-learn components instead.
    predictions = []
    for model in ensemble:
        model.pipeline.fit(Xt, yt)
        predictions.append(model.pipeline.predict(X))
    all_predictions = np.c_[[prediction for prediction in predictions]].T
    return np.mean(all_predictions, axis=1)


def auto_ensemble(cache_dir, metric, y_true, size):
    if isinstance(metric, str):
        metric = string_to_metric(metric)
    models = load_predictions(cache_dir=cache_dir)
    return build_ensemble(models, metric, y_true, n=3, size=size, maximize=True)
