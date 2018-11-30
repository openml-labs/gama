from collections import namedtuple
import os
import pickle
import logging

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import stopit

from gama.genetic_programming.algorithms.metrics import Metric
from gama.utilities.generic.function_dispatcher import FunctionDispatcher

log = logging.getLogger(__name__)
Model = namedtuple("Model", ['name', 'pipeline', 'predictions', 'validation_score'])


class Ensemble(object):

    def __init__(self, metric, y_true,
                 model_library=None, model_library_directory=None,
                 shrink_on_pickle=True, n_jobs=1):
        """
        Either model_library or model_library_directory must be specified.
        If model_library is specified, model_library_directory is ignored.

        :param metric: string or `gama.ea.metrics.Metric`. Metric to optimize the ensemble towards.
        :param y_true: the true labels for the predictions made by the models in the library.
        :param model_library: A list of models from which an ensemble can be built.
        :param model_library_directory: a directory containing results of model evaluations.
        :param shrink_on_pickle: if True, remove memory-intensive attributes that are required during fit,
                                 but not predict, before pickling
        :param n_jobs: the number of jobs to run in parallel when fitting the final ensemble.
        :param label_encoder: a LabelEncoder which can decode the model predictions to desired labels.
        """
        if isinstance(metric, str):
            metric = Metric(metric)
        elif not isinstance(metric, Metric):
            raise ValueError("metric must be specified as string or `gama.ea.metrics.Metric`.")

        if model_library is None and model_library_directory is None:
            raise ValueError("At least one of model_library or model_library_directory must be specified.")

        if model_library is not None and model_library_directory is not None:
            log.warning("model_library_directory will be ignored because model_library is also specified.")

        if not y_true.ndim == 1:
            raise ValueError("Expect y_true to be of shape (N,)")

        self._metric = metric
        self._model_library_directory = model_library_directory
        self._model_library = model_library if model_library is not None else []
        self._shrink_on_pickle = shrink_on_pickle
        self._n_jobs = n_jobs
        self._y_true = y_true
        self._y_score = y_true
        self._prediction_transformation = None

        self._fit_models = None
        self._maximize = True
        self._child_ensembles = []
        self._models = {}

    @property
    def model_library(self):
        if not self._model_library:
            log.debug("Loading model library from disk.")
            self._model_library = load_predictions(self._model_library_directory, self._prediction_transformation)
            log.info("Loaded model library of size {} from disk.".format(len(self._model_library)))

        return self._model_library

    def _total_fit_weights(self):
        return sum([weight for (model, weight) in self._fit_models])

    def _total_model_weights(self):
        return sum([weight for (model, weight) in self._models.values()])

    def _averaged_validation_predictions(self):
        """ Get weighted average of predictions from the self._models on the hillclimb/validation set. """
        weighted_sum_predictions = sum([model.predictions * weight for (model, weight) in self._models.values()])
        return weighted_sum_predictions / self._total_model_weights()

    def build_initial_ensemble(self, n):
        """ Builds an ensemble of n models, based solely on the performance of individual models, not their combined performance.

        :param n: Number of models to include.
        :return: self
        """
        if not n > 0:
            raise ValueError("Ensemble must include at least one model.")
        if self._models:
            log.warning("The ensemble already contained models. Overwriting the ensemble.")
            self._models = {}

        sorted_ensembles = reversed(sorted(self.model_library, key=lambda m: m.validation_score))

        # Since the model library only features unique models, we do not need to check for duplicates here.
        selected_models = list(sorted_ensembles)[:n]
        for model in selected_models:
            self._add_model(model)

        log.debug("Initial ensemble created with score {}".format(self._ensemble_validation_score()))
        return self

    def _add_model(self, model, add_weight=1):
        """ Adds a specific model to the ensemble or increases its weight if it already is contained. """
        model, weight = self._models.pop(model.pipeline, (model, 0))
        self._models[model.pipeline] = (model, weight + add_weight)
        log.info("Assigned a weight of {} to model {}".format(weight + add_weight, model.name))

    def expand_ensemble(self, n):
        """ Adds new models to the ensemble based on earlier given data.

        :param n: Number of models to add to current ensemble.
        :return: self
        """
        if not n > 0:
            raise ValueError("n must be greater than 0.")

        for _ in range(n):
            best_addition_score = -float('inf')
            current_weighted_average = self._averaged_validation_predictions()
            current_total_weight = self._total_model_weights()
            for model in self.model_library:
                if model.validation_score == 0:
                    continue
                candidate_pred = current_weighted_average + \
                                 (model.predictions - current_weighted_average) / (current_total_weight + 1)
                candidate_ensemble_score = self._ensemble_validation_score(candidate_pred)
                if best_addition_score < candidate_ensemble_score:
                    best_addition, best_addition_score = model, candidate_ensemble_score

            self._add_model(best_addition)
            log.debug('Ensemble size {} , best score: {}'.format(self._total_model_weights(), best_addition_score))

        return self

    def fit(self, X, y, timeout=1e6):
        """ Constructs an Ensemble out of the library of models.

        :param X: Data to fit the final selection of models on.
        :param y: Targets corresponding to features X.
        :param timeout: Maximum amount of time in seconds that is allowed in total for fitting pipelines.
                        If this time is exceeded, only pipelines fit until that point are taken into account when making
                        predictions. Starting the parallelization takes roughly 4 seconds by itself.
        :return: self.
        """
        if not self._models:
            raise RuntimeError("You need to call `build` to select models for the ensemble, before fitting them.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

        self._fit_models = []
        fit_dispatcher = FunctionDispatcher(self._n_jobs, fit_and_weight)
        with stopit.ThreadingTimeout(timeout) as c_mgr:
            fit_dispatcher.start()
            for (model, weight) in self._models.values():
                fit_dispatcher.queue_evaluation((model.pipeline, X, y, weight))

            for _ in self._models.values():
                _, output, __ = fit_dispatcher.get_next_result()
                pipeline, weight = output
                if weight > 0:
                    self._fit_models.append((pipeline, weight))

        fit_dispatcher.stop()

        if not c_mgr:
            log.info("Fitting of ensemble stopped early.")

        return self

    def _get_weighted_mean_predictions(self, X, predict_method='predict'):
        weighted_predictions = []
        for (model, weight) in self._fit_models:
            target_prediction = getattr(model, predict_method)(X)
            if self._prediction_transformation:
                target_prediction = self._prediction_transformation(target_prediction)
            weighted_predictions.append(target_prediction * weight)

        return sum(weighted_predictions) / self._total_fit_weights()

    def __str__(self):
        # TODO add internal rank of pipeline
        if not self._models:
            return "Ensemble with no models."
        ensemble_str = "Ensemble of {} unique pipelines.\nW\tScore\tPipeline\n".format(len(self._models))
        for (model, weight) in self._models.values():
            ensemble_str += "{}\t{:.4f}\t{}\n".format(weight, model.validation_score, model.name)
        return ensemble_str

    def __getstate__(self):
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


def load_predictions(cache_dir, prediction_transformation=None):
    models = []
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl'):
            file_name = os.path.join(cache_dir, file)
            if os.stat(file_name).st_size > 0:
                # We check file size, because writing to disk may be interrupted if the process was terminated due
                # to a restart/timeout. I can not find specifications saying that any interrupt of pickle.dump leads
                # to 0-sized files, but in practice this seems to case so far. TODO: Find verification, or fix proper.
                with open(os.path.join(cache_dir, file), 'rb') as fh:
                    pl, predictions, scores = pickle.load(fh)
                if prediction_transformation:
                    predictions = prediction_transformation(predictions)
                models.append(Model(str(pl), pl, predictions, scores))
    return models


def fit_and_weight(args):
    """ Fit the pipeline given the data. Update weight to 0 if fitting fails.

    :return:  pipeline, weight - The same pipeline that was provided as input.
                                 Weight is either the input value of `weight`, if fitting succeeded, or 0 if *any*
                                 exception occurred during fitting.
    """
    pipeline, X, y, weight = args
    try:
        pipeline.fit(X, y)
    except Exception:
        log.warning("Exception when fitting pipeline {} of the ensemble. Assigning weight of 0."
                    .format(pipeline), exc_info=True)
        weight = 0

    return pipeline, weight


class EnsembleClassifier(Ensemble):
    def __init__(self, metric, y_true, label_encoder=None, *args, **kwargs):
        super().__init__(metric, y_true, *args, **kwargs)
        self._label_encoder = label_encoder

        # For metrics that only require class labels, we still want to apply one-hot-encoding to average predictions.
        self._one_hot_encoder = OneHotEncoder(categories='auto').fit(self._y_true.reshape(-1, 1))

        if self._metric.requires_probabilities:
            self._y_score = self._one_hot_encoder.transform(self._y_true.reshape(-1, 1)).toarray()
        else:
            def one_hot_encode_predictions(predictions):
                return self._one_hot_encoder.transform(predictions.reshape(-1, 1))

            self._prediction_transformation = one_hot_encode_predictions

    def _ensemble_validation_score(self, prediction_to_validate=None):
        if prediction_to_validate is None:
            prediction_to_validate = self._averaged_validation_predictions()

        if self._metric.requires_probabilities:
            return self._metric.maximizable_score(self._y_score, prediction_to_validate)
        else:
            # argmax returns (N, 1) matrix, need to squeeze it to (N,) for scoring.
            class_predictions = np.argmax(prediction_to_validate, axis=1).A.ravel()
            return self._metric.maximizable_score(self._y_score, class_predictions)

    def predict(self, X):
        if self._metric.requires_probabilities:
            log.warning('Ensemble was tuned with a class-probabilities metric. '
                        'Using argmax of probabilities, which may not give optimal predictions.')
            class_probabilities = self._get_weighted_mean_predictions(X, 'predict_proba')
        else:
            class_probabilities = self._get_weighted_mean_predictions(X, 'predict').toarray()

        class_predictions = np.argmax(class_probabilities, axis=1)
        if self._label_encoder:
            class_predictions = self._label_encoder.inverse_transform(class_predictions)
        return class_predictions

    def predict_proba(self, X):
        if self._metric.requires_probabilities:
            return self._get_weighted_mean_predictions(X, 'predict_proba')
        else:
            log.warning('Ensemble was tuned with a class label predictions metric, not probabilities. '
                        'Using weighted mean of class predictions.')
            return self._get_weighted_mean_predictions(X, 'predict').toarray()


class EnsembleRegressor(Ensemble):
    def _ensemble_validation_score(self, prediction_to_validate=None):
        if prediction_to_validate is None:
            prediction_to_validate = self._averaged_validation_predictions()
        return self._metric.maximizable_score(self._y_score, prediction_to_validate)

    def predict(self, X):
        return self._get_weighted_mean_predictions(X)
