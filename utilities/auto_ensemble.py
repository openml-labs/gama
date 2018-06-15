from collections import namedtuple, Counter
import os
import pickle
import logging
from joblib import Parallel, delayed

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..ea.evaluation import string_to_metric, evaluate, Metric

log = logging.getLogger(__name__)
Model = namedtuple("Model", ['name', 'pipeline', 'predictions', 'validation_score'])


class Ensemble(object):

    def __init__(self, metric, y_true,
                 model_library=None, model_library_directory=None,
                 shrink_on_pickle=True, n_jobs=1):
        """
        Either model_library or model_library_directory must be specified.
        If model_library is specified, model_library_directory is ignored.

        :param metric: metric to optimize the ensemble towards.
        :param y_true: the true labels for the predictions made by the models in the library.
        :param model_library: A list of models from which an ensemble can be built.
        :param model_library_directory: a directory containing results of model evaluations.
        :param shrink_on_pickle: if True, remove memory-intensive attributes that are required during fit,
                                 but not predict, before pickling
        :param n_jobs: the number of jobs to run in parallel when fitting the final ensemble.
        """
        if isinstance(metric, str):
            metric = string_to_metric(metric)

        if model_library is None and model_library_directory is None:
            raise ValueError("At least one of model_library or model_library_directory must be specified.")

        if model_library is not None and model_library_directory is not None:
            log.warning("model_library_directory will be ignored because model_library is also specified.")

        self._metric = metric
        self._y_true = y_true
        self._model_library_directory = model_library_directory
        self._model_library = model_library if model_library is not None else []
        self._shrink_on_pickle = shrink_on_pickle
        self._n_jobs = n_jobs

        self._fit_models = None
        self._maximize = True
        self._child_ensembles = []
        self._child_ensemble_model_fraction = 0.3
        self._models = []

    @property
    def model_library(self):
        if not self._model_library:
            log.debug("Loading model library from disk.")
            self._model_library = load_predictions(self._model_library_directory)

            # This is a work-around because there can be multiple of the same model in the model library.
            # This causes extra workload in non-bagging case, and breaks the bagging case (because a uniform sample
            # is no longer guaranteed).
            # TODO: Fix cause of this issue instead.
            unique_model_names = {model.name for model in self._model_library}
            self._model_library = [[model for model in self._model_library if model.name == model_name][0]
                                   for model_name in unique_model_names]
        return self._model_library

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

    def build_(self, n_ensembles, start_size, total_size):

        for i in range(n_ensembles):
            log.debug("Constructing ensemble {}/{}.".format(i, n_ensembles))

            subset_n = int(self._child_ensemble_model_fraction * len(self.model_library))
            model_subset_indices = np.random.choice(range(len(self.model_library)), size=subset_n, replace=False)
            model_subset = [self.model_library[idx] for idx in model_subset_indices]

            child_ensemble = Ensemble(self._metric, self._y_true, model_library=model_subset)
            child_ensemble.build_initial_ensemble(start_size)
            child_ensemble.add_models(total_size - start_size)

            self._child_ensembles.append(child_ensemble)
            # we combine all the weights of child models, this way we don't have to fit each individual ensemble.

        self._models = [model for child_ensemble in self._child_ensembles for model in child_ensemble._models]

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

        #sorted_ensembles = sorted(self.model_library, key=lambda m: m.validation_score)
        sorted_ensembles = sorted(self.model_library, key=lambda m: evaluate(self._metric, self._y_true, m.predictions))
        if self._maximize:
            sorted_ensembles = reversed(sorted_ensembles)

        # Since the model library only features unique models, we do not need to check for duplicates here.
        selected_models = list(sorted_ensembles)[:n]
        self._models = {m.name: (m, 1) for m in selected_models}
        log.debug("Initial ensemble created with score {}".format(
                  evaluate(self._metric, self._y_true, self._averaged_validation_predictions())))
        return self

    def _total_model_weights(self):
        return sum([weight for (model, weight) in self._models.values()])

    def _averaged_validation_predictions(self):
        """ Get weighted average of predictions from the self._models on the hillclimb/validation set. """
        weighted_predictions = np.stack([model.predictions * weight for (model, weight) in self._models.values()])
        return np.sum(weighted_predictions, axis=0) / self._total_model_weights()

    def evaluate_ensemble(ensemble, metric, y_true):
        """ Evaluates the ensemble according to the metric.

        Currently assumes a single prediction value (e.g. numeric response or positive class probability).
        """
        all_predictions = np.stack([model.predictions for model in ensemble])
        average_predictions = np.mean(all_predictions, axis=0)
        return evaluate(metric, y_true, average_predictions)

    def _add_model(self, model):
        """ Adds a specific model to the ensemble. """
        if model.name in self._models:
            model, weight = self._models[model.name]
            self._models[model.name] = (model, weight + 1)
        else:
            self._models[model.name] = (model, 1)

    def add_models(self, n):
        """ Adds new models to the ensemble based on earlier given data.

        :param n: Number of models to add to current ensemble.
        :return: self
        """
        if not n > 0:
            raise ValueError("n must be greater than 0.")

        for _ in range(n):
            best_addition_score = -float('inf') if self._maximize else float('inf')
            current_weighted_average = self._averaged_validation_predictions()
            current_total_weight = self._total_model_weights()
            for model in self.model_library:
                if model.validation_score == 0:
                    continue
                candidate_pred = current_weighted_average + \
                                 (model.predictions - current_weighted_average) / (current_total_weight + 1)
                candidate_ensemble_score = evaluate(self._metric, self._y_true, candidate_pred)
                if ((self._maximize and best_addition_score < candidate_ensemble_score) or
                        (not self._maximize and best_addition_score > candidate_ensemble_score)):
                    best_addition, best_addition_score = model, candidate_ensemble_score

            self._add_model(best_addition)
            log.debug('Ensemble size {} , best score: {}'.format(self._total_model_weights(), best_addition_score))
            log.debug(str(self))

        return self

    def fit(self, X, y):
        """ Constructs an Ensemble out of the library of models.

        :param X: Data to fit the final selection of models on.
        :param y: Targets corresponding to features X.
        :return: self.
        """
        if not self._models:
            raise RuntimeError("You need to call `build` to select models for the ensemble, before fitting them.")


        self._fit_models = Parallel(n_jobs=self._n_jobs)(delayed(fit_and_weight)
                                   (model.pipeline, X, y, weight) for (model, weight) in self._models.values())

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
        for (model, weight) in self._fit_models:
            if hasattr(model, 'predict_proba'):
                predictions.append(model.predict_proba(X) * weight)
            else:
                target_prediction = model.predict(X)
                if self._metric.is_classification:
                    ohe_prediction = OneHotEncoder(len(set(self._y_true))).fit_transform(target_prediction.reshape(-1, 1)).todense()
                    predictions.append(np.array(ohe_prediction) * weight)
                elif self._metric.is_regression:
                    predictions.append(target_prediction * weight)
                else:
                    raise NotImplemented('Unknown task type for ensemble.')

        if len(self._fit_models) == 1:
            return predictions[0]
        else:
            all_predictions = np.stack(predictions)
            return np.sum(all_predictions, axis=0) / self._total_model_weights()

    def __str__(self):
        # TODO add internal score and rank of pipeline
        if not self._models:
            return "Ensemble with no models."
        ensemble_str = "Ensemble of {} unique pipelines.\nW\tScore\tPipeline\n".format(len(self._models))
        for (model, weight) in self._models.values():
            ensemble_str += "{}\t{:.4f}\t{}\n".format(weight, model.validation_score, model.name)
        return ensemble_str

    def __getstate__(self):
        # TODO: Fix properly. Workaround for unpicklable local 'neg' functions.
        if 'neg' in self._metric.name:
            name, fn, *rest = self._metric
            self._metric = Metric(name, None, *rest)

        if self._shrink_on_pickle:
            log.info('Shrinking before pickle because shrink_on_pickle is True.'
                     'Removing anything that is not needed for predict-functionality.'
                     'Functionality to expand ensemble after unpickle is not available.')
            self._models = None
            self._model_library = None
            self._child_ensembles = None
            # self._y_true can not be removed as it is needed to ensure proper dimensionality of predictions
            # alternatively, one could just save the number of classes instead.

        return self.__dict__.copy()


def load_predictions(cache_dir, argmax_pred=False):
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

            models.append(Model(str(pl), pl, predictions, score))
    return models


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

def fit_and_weight(pipeline, X, y, weight):
    pipeline.fit(X, y)
    return (pipeline, weight)