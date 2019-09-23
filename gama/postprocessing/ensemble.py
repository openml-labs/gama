from collections import namedtuple
import os
import pickle
import logging
import time
from typing import Optional, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import stopit

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.utilities.metrics import Metric, MetricType
from gama.utilities.generic.async_evaluator import AsyncEvaluator

log = logging.getLogger(__name__)
Model = namedtuple("Model", ['name', 'pipeline', 'predictions', 'validation_score'])


class EnsemblePostProcessing(BasePostProcessing):

    def __init__(self, time_fraction: float = 0.3, ensemble_size: int = 25):
        """ Ensemble construction per Caruana et al.

        Parameters
        ----------
        time_fraction: float (default=0.3)
            Fraction of total time reserved for Ensemble building.
        ensemble_size: int (default=25)
            Total number of models in the ensemble.
            When a single model is chosen more than once, it will increase its weight in the ensemble and
            *does* count towards this maximum.
        """
        super().__init__(time_fraction)
        self.ensemble_size = ensemble_size
        self.metric = None
        self.cache = None

    def dynamic_defaults(self, gama: 'Gama'):
        self.metric = gama._metrics[0]
        self.cache = gama._cache_dir

    def post_process(self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]) -> 'model':
        return build_fit_ensemble(x, y, self.ensemble_size, timeout, self.metric, self.cache)


class Ensemble(object):

    def __init__(self, metric, y: pd.DataFrame,
                 model_library=None, model_library_directory=None,
                 shrink_on_pickle=True):
        """
        Either model_library or model_library_directory must be specified.
        If model_library is specified, model_library_directory is ignored.

        Parameters
        ----------
        metric: string or Metric
            Metric to optimize the ensemble towards.
        y: pandas.DataFrame
            True labels for the predictions made by the models in the library.
        model_library:
            A list of models from which an ensemble can be built.
        model_library_directory: str
            a directory containing results of model evaluations.
        shrink_on_pickle: bool (default=True)
            If True, remove memory-intensive attributes that are required before pickling.
            When unpickled, the model can be used to create predictions, but the ensemble can't be changed.
        """
        if isinstance(metric, str):
            metric = Metric(metric)
        elif not isinstance(metric, Metric):
            raise ValueError("metric must be specified as string or `gama.ea.metrics.Metric`.")

        if model_library is None and model_library_directory is None:
            raise ValueError("At least one of model_library or model_library_directory must be specified.")

        if model_library is not None and model_library_directory is not None:
            log.warning("model_library_directory will be ignored because model_library is also specified.")

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError(f"`y_true` must be of type pandas.DataFrame or pandas.Series but is {type(y)}.")

        self._metric = metric
        self._model_library_directory = model_library_directory
        self._model_library = model_library if model_library is not None else []
        self._shrink_on_pickle = shrink_on_pickle
        self._y = y
        self._prediction_transformation = None

        self._internal_score = None
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

    def _ensemble_validation_score(self, prediction_to_validate=None):
        raise NotImplementedError("Must be implemented by child class.")

    def _total_fit_weights(self):
        return sum([weight for (model, weight) in self._fit_models])

    def _total_model_weights(self):
        return sum([weight for (model, weight) in self._models.values()])

    def _averaged_validation_predictions(self):
        """ Get weighted average of predictions from the self._models on the hillclimb/validation set. """
        weighted_sum_predictions = sum([model.predictions * weight for (model, weight) in self._models.values()])
        return weighted_sum_predictions / self._total_model_weights()

    def build_initial_ensemble(self, n: int):
        """ Builds an ensemble of n models, based solely on the performance of individual models, not their combined performance.

        Parameters
        ----------
        n: int
            Number of models to include.
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

    def _add_model(self, model, add_weight=1):
        """ Adds a specific model to the ensemble or increases its weight if it already is contained. """
        model, weight = self._models.pop(model.pipeline, (model, 0))
        self._models[model.pipeline] = (model, weight + add_weight)
        log.debug("Assigned a weight of {} to model {}".format(weight + add_weight, model.name))

    def expand_ensemble(self, n: int):
        """ Adds new models to the ensemble based on earlier given data.

        Parameters
        ----------
        n: int
            Number of models to add to current ensemble.
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
            self._internal_score = best_addition_score
            log.info('Ensemble size {} , best score: {}'.format(self._total_model_weights(), best_addition_score))

    def fit(self, X, y, timeout=1e6):
        """ Constructs an Ensemble out of the library of models.

        Parameters
        ----------
        X:
            Data to fit the final selection of models on.
        y:
            Targets corresponding to features X.
        timeout: int (default=1e6)
            Maximum amount of time in seconds that is allowed in total for fitting pipelines.
            If this time is exceeded, only pipelines fit until that point are taken into account when making
            predictions. Starting the parallelization takes roughly 4 seconds by itself.
        """
        if not self._models:
            raise RuntimeError("You need to call `build` to select models for the ensemble, before fitting them.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

        self._fit_models = []
        futures = set()
        with stopit.ThreadingTimeout(timeout) as c_mgr, AsyncEvaluator() as async_:
            for (model, weight) in self._models.values():
                futures.add(async_.submit(fit_and_weight, (model.pipeline, X, y, weight)))

            for _ in self._models.values():
                future = async_.wait_next()
                pipeline, weight = future.result
                if weight > 0:
                    self._fit_models.append((pipeline, weight))

        if not c_mgr:
            log.info("Fitting of ensemble stopped early.")

    def _get_weighted_mean_predictions(self, X, predict_method='predict'):
        weighted_predictions = []
        for (model, weight) in self._fit_models:
            target_prediction = getattr(model, predict_method)(X)
            if self._prediction_transformation:
                target_prediction = self._prediction_transformation(target_prediction)
            weighted_predictions.append(target_prediction * weight)

        return sum(weighted_predictions) / self._total_fit_weights()

    def __str__(self):
        if not self._models:
            return "Ensemble with no models."
        ensemble_str = "Ensemble of {} unique pipelines.\nR\tW\tScore\tPipeline\n".format(len(self._models))
        for i, (model, weight) in enumerate(sorted(self._models.values(), key=lambda x: x[0].validation_score)):
            ensemble_str += "{}\t{}\t{:.4f}\t{}\n".format(i, weight, model.validation_score[0], model.name)
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
                # to 0-sized files, but in practice this seems to case so far.
                with open(os.path.join(cache_dir, file), 'rb') as fh:
                    pl, predictions, scores = pickle.load(fh)
                if prediction_transformation:
                    predictions = prediction_transformation(predictions)
                models.append(Model(str(pl), pl, predictions, scores))
    return models


def fit_and_weight(args):
    """ Fit the pipeline given the data. Update weight to 0 if fitting fails.

    Parameters
    ----------
    args: Tuple
        Expected Tuple of [Pipeline, X, y, weight].

    Returns
    -------
    pipeline
        The same pipeline that was provided as input.
    weight
        If fitting succeeded, return the input weight.
        If *any* exception occurred during fitting, weight is 0.
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
        y_as_squeezed_array = y_true.values.reshape(-1, 1)
        self._one_hot_encoder = OneHotEncoder(categories='auto').fit(y_as_squeezed_array)

        if self._metric.requires_probabilities:
            self._y = self._one_hot_encoder.transform(y_as_squeezed_array).toarray()
        else:
            def one_hot_encode_predictions(predictions):
                return self._one_hot_encoder.transform(predictions.reshape(-1, 1))

            self._prediction_transformation = one_hot_encode_predictions

    def _ensemble_validation_score(self, prediction_to_validate=None):
        if prediction_to_validate is None:
            prediction_to_validate = self._averaged_validation_predictions()

        if self._metric.requires_probabilities:
            return self._metric.maximizable_score(self._y, prediction_to_validate)
        else:
            # argmax returns (N, 1) matrix, need to squeeze it to (N,) for scoring.
            class_predictions = self._one_hot_encoder.inverse_transform(prediction_to_validate.toarray())
            return self._metric.maximizable_score(self._y, class_predictions)

    def predict(self, X):
        if self._metric.requires_probabilities:
            log.warning('Ensemble was tuned with a class-probabilities metric. '
                        'Using argmax of probabilities, which may not give optimal predictions.')
            class_probabilities = self._get_weighted_mean_predictions(X, 'predict_proba')
        else:
            class_probabilities = self._get_weighted_mean_predictions(X, 'predict').toarray()

        class_predictions = self._one_hot_encoder.inverse_transform(class_probabilities)
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
        return self._metric.maximizable_score(self._y, prediction_to_validate)

    def predict(self, X):
        return self._get_weighted_mean_predictions(X)


def build_fit_ensemble(x, y, ensemble_size: int, timeout: float,
                       metric: Metric, cache: str, encoder: Optional[object]=None) -> Ensemble:
    """ Construct an Ensemble of models from cache, optimizing for metric and fit to (x, y). """
    start_build = time.time()

    log.debug('Building ensemble.')
    if metric.task_type == MetricType.CLASSIFICATION:
        ensemble = EnsembleClassifier(metric, y, model_library_directory=cache)
        ensemble._label_encoder = encoder
    elif metric.task_type == MetricType.REGRESSION:
        ensemble = EnsembleRegressor(metric, y, model_library_directory=cache)
    else:
        raise ValueError(f"Unknown metric task type {metric.task_type}")

    try:
        # Starting with more models in the ensemble should help against overfitting, but depending on the total
        # ensemble size, it might leave too little room to calibrate the weights or add new models. So we have
        # some adaptive defaults (for now).
        if ensemble_size <= 10:
            ensemble.build_initial_ensemble(1)
        else:
            ensemble.build_initial_ensemble(10)

        remainder = ensemble_size - ensemble._total_model_weights()
        if remainder > 0:
            ensemble.expand_ensemble(remainder)

        build_time = time.time() - start_build
        timeout = timeout - build_time
        log.info(f'Building ensemble took {build_time}s. Fitting ensemble with timeout {timeout}s.')

        ensemble.fit(x, y, timeout)
    except Exception as e:
        log.warning(f"Error during auto ensemble: {e}", exc_info=True)

    return ensemble
