"""

"""

from collections import namedtuple, Counter
import os
import pickle
import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..ea.evaluation import string_to_metric, evaluate, Metric

log = logging.getLogger(__name__)
Model = namedtuple("Model", ['name', 'pipeline', 'predictions'])


class Ensemble(object):

    def __init__(self, model_library_directory, metric, y_true):
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
        self._models = []
        self._fit_models = None
        self._maximize = True

    def build(self, n_models_in_ensemble, start_size=1):
        """ Constructs an ensemble out of a library of models.

        :param n_models_in_ensemble: The number of models to include in the ensemble.
        :param start_size: The number of n best models to include automatically.
        :return: self
        """
        if start_size > n_models_in_ensemble:
            raise ValueError("`n_models_in_ensemble` cannot be smaller than start_size specified."
                             "Values were respectively {} and {}".format(n_models_in_ensemble, start_size))

        models = load_predictions(self._model_library_directory)
        self._models = build_ensemble(models, self._metric, self._y_true,
                                      start_size=start_size, end_size=n_models_in_ensemble)
        return self

    def build_initial_ensemble(self, n):
        """ Builds an ensemble of n models, based solely on the performance of individual models, not their combined performance.

        :param n: Number of models to include.
        :return: self
        """
        pass
        if not n > 0:
            raise ValueError("Ensemble must include at least one model.")
        if self._models:
            log.warning("The ensemble already contained models. Overwriting the ensemble.")

        models = load_predictions(self._model_library_directory)
        sorted_ensembles = sorted(models, key=lambda m: evaluate(self._metric, self._y_true, m.predictions))
        if self._maximize:
            sorted_ensembles = reversed(sorted_ensembles)
        self._models = list(sorted_ensembles)[:n]
        return self

    def add_models(self, n):
        """ Adds new models to the ensemble based on earlier given data.

        :param n: Number of models to add to current ensemble.
        :return: self
        """
        if not n > 0:
            raise ValueError("n must be greater than 0.")

        models = load_predictions(self._model_library_directory)
        ensemble = self._models
        best_ensemble = ensemble

        for _ in range(n):
            best_ensemble_score = -float('inf') if self._maximize else float('inf')
            for model in models:
                candidate_ensemble = ensemble + [model]
                candidate_ensemble_score = evaluate_ensemble(candidate_ensemble, self._metric, self._y_true)
                if ((self._maximize and best_ensemble_score < candidate_ensemble_score) or
                        (not self._maximize and best_ensemble_score > candidate_ensemble_score)):
                    best_ensemble, best_ensemble_score = candidate_ensemble, candidate_ensemble_score
            ensemble = best_ensemble
            log.debug('Ensemble size {} , best score: {}'.format(len(ensemble), best_ensemble_score))

        self._models = ensemble
        return self

    def fit(self, X, y):
        """ Constructs an Ensemble out of the library of models.

        :param X: Data to fit the final selection of models on.
        :param y: Targets corresponding to features X.
        :return: self.
        """
        if not self._models:
            raise RuntimeError("You need to call `build` to select models for the ensemble, before fitting them.")

        self._fit_models = [model.pipeline.fit(X, y) for model in self._models]

        return self

    def predict(self, X):
        if self._metric.is_classification:
            predictions = np.argmax(self.predict_proba(X), axis=1)
            return np.squeeze(predictions)
        elif self._metric.is_regression:
            return self.predict_proba(X)
        else:
            raise NotImplemented('Unknown task type for ensemble.')

    def predict_proba(self, X):
        predictions = []
        for model in self._fit_models:
            if hasattr(model, 'predict_proba'):
                predictions.append(model.predict_proba(X))
            else:
                target_prediction = model.predict(X)
                if self._metric.is_classification:
                    ohe_prediction = OneHotEncoder().fit_transform(target_prediction.reshape(-1, 1)).todense()
                    predictions.append(np.array(ohe_prediction))
                elif self._metric.is_regression:
                    predictions.append(target_prediction)
                else:
                    raise NotImplemented('Unknown task type for ensemble.')

        if len(self._fit_models) == 1:
            return predictions[0]
        else:
            all_predictions = np.stack(predictions)
            return np.mean(all_predictions, axis=0)

    def __str__(self):
        # TODO add internal score and rank of pipeline
        if not self._models:
            return "Ensemble with no models."
        components = Counter([m.name for m in self._models])
        ensemble_str = "Ensemble of {} unique pipelines.\nW\tPipeline\n".format(len(components))
        for pipeline, count in components.items():
            ensemble_str += "{}\t{}\n".format(count, pipeline)
        return ensemble_str

    def __getstate__(self):
        # TODO: Fix properly. Workaround for unpicklable local 'neg' functions.
        if 'neg' in self._metric.name:
            name, fn, *rest = self._metric
            self._metric = Metric(name, None, *rest)
        # capture what is normally pickled
        state = self.__dict__.copy()
        # what we return here will be stored in the pickle
        return state


def load_predictions(cache_dir, argmax_pred=True):
    models = []
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(cache_dir, file), 'rb') as fh:
                pl, predictions, score = pickle.load(fh)
                predictions = np.array(predictions)
                if argmax_pred:
                    hard_predictions = np.argmax(predictions, axis=1)
                    positions = zip(range(len(hard_predictions)), hard_predictions)
                    ind_predictions = np.zeros_like(predictions)
                    for pos in positions:
                        ind_predictions[pos] = 1
                    predictions = ind_predictions

            models.append(Model(str(pl), pl, predictions))
    return models


def evaluate_ensemble(ensemble, metric, y_true):
    """ Evaluates the ensemble according to the metric.

    Currently assumes a single prediction value (e.g. numeric response or positive class probability).
    """
    all_predictions = np.stack([model.predictions for model in ensemble])
    average_predictions = np.mean(all_predictions, axis=0)
    return evaluate(metric, y_true, average_predictions)


def build_ensemble(models, metric, y_true, start_size=0, end_size=5, maximize=True, consider_top_n=1000):
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
    sorted_ensembles = list(sorted_ensembles)[:consider_top_n]
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
