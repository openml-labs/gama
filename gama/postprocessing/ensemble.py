import uuid
import copy
import logging
import random
import time
from typing import Optional, List, TYPE_CHECKING, Dict, Tuple, Sequence

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.utilities.evaluation_library import EvaluationLibrary, Evaluation
from gama.utilities.export import (
    imports_and_steps_for_individual,
    format_import,
    format_pipeline,
    transformers_to_str,
)
from gama.utilities.metrics import Metric, MetricType


if TYPE_CHECKING:
    from gama.gama import Gama

log = logging.getLogger(__name__)


class EnsemblePostProcessing(BasePostProcessing):
    def __init__(
        self,
        time_fraction: float = 0.3,
        ensemble_size: Optional[int] = 25,
        hillclimb_size: Optional[int] = 10_000,
        max_models: Optional[int] = 200,
    ):
        """ Ensemble construction per Caruana et al.

        Parameters
        ----------
        time_fraction: float (default=0.3)
            Fraction of total time reserved for Ensemble building.
        ensemble_size: int, optional (default=25)
            Total number of models in the ensemble.
            When a single model is chosen more than once, it will increase its weight
            in the ensemble and *does* count towards this maximum.
        hillclimb_size: int, optional (default=10_000)
            Number of predictions that are used to determine the ensemble score
            during hillclimbing. If `None`, use all.
        max_models: int, optional (default=200)
            Only consider the best `max_models` number of models. If `None`, use all.
            Consequently also sets the max number of unique models in the ensemble.
        """
        super().__init__(time_fraction)
        self._hyperparameters = dict(
            ensemble_size=(ensemble_size, 25),
            metric=(None, None),
            evaluation_library=(None, None),
            hillclimb_size=(hillclimb_size, 10_000),
            max_models=(max_models, 200),
        )
        self._ensemble: Optional[Ensemble] = None

    def dynamic_defaults(self, gama: "Gama"):
        self._overwrite_hyperparameter_default("metric", gama._metrics[0])
        self._overwrite_hyperparameter_default(
            "evaluation_library", gama._evaluation_library
        )

    def post_process(
        self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]
    ) -> object:
        self._ensemble = build_fit_ensemble(
            x,
            y,
            self.hyperparameters["ensemble_size"],
            timeout,
            self.hyperparameters["metric"],
            self.hyperparameters["evaluation_library"],
        )
        return self._ensemble

    def to_code(
        self, preprocessing: Sequence[Tuple[str, TransformerMixin]] = None
    ) -> str:
        if isinstance(self._ensemble, EnsembleClassifier):
            voter = "VotingClassifier"
        elif isinstance(self._ensemble, EnsembleRegressor):
            voter = "VotingRegressor"
        else:
            raise RuntimeError(f"Can't export ensemble of type {type(self._ensemble)}.")

        imports = {
            f"from sklearn.ensemble import {voter}",
            "from sklearn.pipeline import Pipeline",
        }

        pipelines = []
        for i, (model, weight) in enumerate(self._ensemble._models.values()):
            ind_imports, steps = imports_and_steps_for_individual(model.individual)
            imports = imports.union(ind_imports)
            pipeline_name = f"pipeline_{i}"
            pipelines.append(format_pipeline(steps, name=pipeline_name))

        estimators = ",".join([f"('{i}', pipeline_{i})" for i in range(len(pipelines))])
        weights = [weight for _, weight in self._ensemble._models.values()]

        if isinstance(self._ensemble, EnsembleClassifier):
            if self._ensemble._metric.requires_probabilities:
                voting = ",'soft'"
            else:
                voting = ", 'hard'"
        else:
            voting = ""  # This parameter does not exist for VotingRegressor

        if preprocessing is not None:
            imports = imports.union({format_import(t) for _, t in preprocessing})

        script = (
            "\n".join(sorted(imports))
            + "\n\n"
            + "\n\n".join(pipelines)
            + "\n"
            + f"ensemble = {voter}([{estimators}]{voting},{weights})\n"
        )
        if preprocessing is not None:
            trans_strs = transformers_to_str([t for _, t in preprocessing])
            names = [name for name, _ in preprocessing]
            steps = list(zip(names, trans_strs))
            script += format_pipeline(steps + [("ensemble", "ensemble")])
        return script


class Ensemble(object):
    def __init__(
        self,
        metric,
        y: pd.DataFrame,
        evaluation_library: EvaluationLibrary = None,
        shrink_on_pickle=True,
        downsample_to: Optional[int] = 10_000,
        use_top_n_only: Optional[int] = 200,
    ):
        """
        Either model_library or model_library_directory must be specified.
        If model_library is specified, model_library_directory is ignored.

        Parameters
        ----------
        metric: string or Metric
            Metric to optimize the ensemble towards.
        y: pandas.DataFrame
            True labels for the predictions made by the models in the library.
        evaluation_library: `gama.utilities.evaluation_library.EvaluationLibrary`
            A list of models from which an ensemble can be built.
        shrink_on_pickle: bool (default=True)
            If True, remove memory-intensive attributes that are required before pickle.
            When unpickled, the model can be used to create predictions,
            but the ensemble can't be changed.
        """
        if isinstance(metric, str):
            metric = Metric(metric)
        elif not isinstance(metric, Metric):
            raise ValueError(
                "metric must be specified as string or `gama.ea.metrics.Metric`."
            )

        if evaluation_library is None:
            raise ValueError(
                "`evaluation_library` is None but must be EvaluationLibrary."
            )
        elif not isinstance(evaluation_library, EvaluationLibrary):
            raise TypeError(
                "`evaluation_library` must be of type "
                "gama.utilities.evaluation_library.EvaluationLibrary."
            )

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError(f"y_true must be pd.DataFrame or pd.Series, is {type(y)}.")

        self._metric = metric
        self.evaluation_library = evaluation_library
        self._model_library: List[Evaluation] = []
        self._use_top_n_only = use_top_n_only
        self._shrink_on_pickle = shrink_on_pickle
        self._prediction_transformation = None

        if self.evaluation_library._sample is not None:
            # If the library stores sampled predictions, we match that first.
            y = y.iloc[self.evaluation_library._sample]

        # Then apply even more sampling if requested.
        if downsample_to is None or downsample_to >= len(y):
            if downsample_to is not None:
                log.info(f"Not downsampling because only {len(y)} samples were stored.")
            self._y = y
            self._prediction_sample = None
        else:
            log.info(f"Downsampling as training data exceeds {downsample_to} samples.")
            self._prediction_sample = random.sample(range(len(y)), downsample_to)
            self._y = y.iloc[self._prediction_sample]

        self._internal_score = -float("inf")
        self._fit_models = None
        self._maximize = True
        self._models: Dict[uuid.UUID, Tuple[Evaluation, int]] = {}

    @property
    def model_library(self):
        if not self._model_library:
            self._model_library = []
            for evaluation in self.evaluation_library.n_best(self._use_top_n_only):
                predictions = evaluation.predictions
                if self._prediction_transformation:
                    predictions = self._prediction_transformation(predictions)
                if self._prediction_sample:
                    predictions = predictions[self._prediction_sample]

                e = copy.copy(evaluation)
                e._predictions = predictions
                self._model_library.append(e)

        return self._model_library

    def _ensemble_validation_score(self, prediction_to_validate=None):
        raise NotImplementedError("Must be implemented by child class.")

    def _total_fit_weights(self):
        return sum([weight for (model, weight) in self._fit_models])

    def _total_model_weights(self):
        return sum([weight for (model, weight) in self._models.values()])

    def _averaged_validation_predictions(self):
        """ Weighted average of predictions of current models on the hillclimb set. """
        weighted_sum_predictions = sum(
            [model.predictions * weight for (model, weight) in self._models.values()]
        )
        return weighted_sum_predictions / self._total_model_weights()

    def build_initial_ensemble(self, n: int):
        """ Add top n models in EvaluationLibrary to the ensemble.

        Parameters
        ----------
        n: int
            Number of models to include.
        """
        if not n > 0:
            raise ValueError("Ensemble must include at least one model.")
        if self._models:
            log.warning(
                "The ensemble already contained models. Overwriting the ensemble."
            )
            self._models = {}

        # Since the model library only features unique models,
        # we do not need to check for duplicates here.
        for model in self.model_library[:n]:
            self._add_model(model)

        log.debug(
            "Initial ensemble created with score {}".format(
                self._ensemble_validation_score()
            )
        )

    def _add_model(self, model, add_weight=1):
        """ Add a specific model to the ensemble or increases its weight. """
        model, weight = self._models.pop(model.individual._id, (model, 0))
        new_weight = weight + add_weight
        self._models[model.individual._id] = (model, new_weight)
        log.debug(f"Weight {model.individual.short_name('>')} set to {new_weight}.")

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
            best_addition_score = -float("inf")
            current_weighted_average = self._averaged_validation_predictions()
            current_total_weight = self._total_model_weights()
            for model in self.model_library:
                if model.score == 0:
                    continue
                candidate_pred = current_weighted_average + (
                    model.predictions - current_weighted_average
                ) / (current_total_weight + 1)
                candidate_ensemble_score = self._ensemble_validation_score(
                    candidate_pred
                )
                if best_addition_score < candidate_ensemble_score:
                    best_addition, best_addition_score = model, candidate_ensemble_score

            self._add_model(best_addition)
            self._internal_score = best_addition_score
            log.info(
                "Ensemble size {} , best score: {}".format(
                    self._total_model_weights(), best_addition_score
                )
            )

    def fit(self, x, y, timeout=1e6):
        """ Constructs an Ensemble out of the library of models.

        Parameters
        ----------
        x:
            Data to fit the final selection of models on.
        y:
            Targets corresponding to features x.
        timeout: int (default=1e6)
            Maximum amount of time in seconds that is allowed for fitting pipelines.
            If this time is exceeded, only pipelines fit until that point are taken
            into account when making predictions.
            Starting the parallelization takes roughly 4 seconds by itself.
        """
        if not self._models:
            raise RuntimeError(
                "You need to call `build` to select models before fitting them."
            )
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0.")

        self._fit_models = [
            (estimator, weight)
            for (model, weight) in self._models.values()
            for estimator in model.estimators
        ]
        # for (model, weight) in self._models.values():

        # self._fit_models = []
        # futures = set()
        # with stopit.ThreadingTimeout(timeout) as c_mgr, AsyncEvaluator() as async_:
        #     for (model, weight) in self._models.values():
        #         futures.add(
        #           async_.submit(fit_and_weight, (model.pipeline, X, y, weight))
        #           )
        #
        #     for _ in self._models.values():
        #         future = async_.wait_next()
        #         pipeline, weight = future.result
        #         if weight > 0:
        #             self._fit_models.append((pipeline, weight))
        #
        # if not c_mgr:
        #     log.info("Fitting of ensemble stopped early.")

    def _get_weighted_mean_predictions(self, X, predict_method="predict"):
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
        _str_ = f"Ensemble of {len(self._models)} pipelines.\nR\tW\tScore\tPipeline\n"
        models = sorted(self._models.values(), key=lambda x: x[0].validation_score)
        for i, (model, weight) in enumerate(models):
            _str_ += f"{i}\t{weight}\t{model.validation_score[0]:.4f}\t{model.name}\n"
        return _str_

    def __getstate__(self):
        if self._shrink_on_pickle:
            log.info(
                "Shrinking before pickle because shrink_on_pickle is True."
                "Removing anything that is not needed for predict-functionality."
                "Functionality to expand ensemble after unpickle is not available."
            )
            self._models = None
            self._model_library = None
            # self._y_true can not be removed as it is needed to ensure proper
            # dimensionality of predictions.
            # Alternatively, one could just save the number of classes instead.

        return self.__dict__.copy()


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
        log.warning(f"Exception fitting {pipeline}. Set weight to 0.", exc_info=True)
        weight = 0

    return pipeline, weight


class EnsembleClassifier(Ensemble):
    def __init__(self, metric, y_true, label_encoder=None, *args, **kwargs):
        super().__init__(metric, y_true, *args, **kwargs)
        self._label_encoder = label_encoder

        # For metrics that only require class labels,
        # we still want to apply one-hot-encoding to average predictions.
        y_as_squeezed_array = y_true.to_numpy().reshape(-1, 1)
        self._one_hot_encoder = OneHotEncoder(categories="auto")
        self._one_hot_encoder.fit(y_as_squeezed_array)

        if self._metric.requires_probabilities:
            self._y = self._one_hot_encoder.transform(
                self._y.to_numpy().reshape(-1, 1)
            ).toarray()
            if self._prediction_sample is not None:
                self._y = self._y[self._prediction_sample]
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
            class_predictions = self._one_hot_encoder.inverse_transform(
                prediction_to_validate.toarray()
            )
            return self._metric.maximizable_score(self._y, class_predictions)

    def predict(self, X):
        if self._metric.requires_probabilities:
            log.warning(
                "Ensemble was tuned with a class-probabilities metric. "
                "Using argmax of probabilities, which may not give optimal predictions."
            )
            class_probabilities = self._get_weighted_mean_predictions(
                X, "predict_proba"
            )
        else:
            class_probabilities = self._get_weighted_mean_predictions(
                X, "predict"
            ).toarray()

        class_predictions = self._one_hot_encoder.inverse_transform(class_probabilities)
        if self._label_encoder:
            class_predictions = self._label_encoder.inverse_transform(class_predictions)

        return class_predictions.ravel()

    def predict_proba(self, X):
        if self._metric.requires_probabilities:
            return self._get_weighted_mean_predictions(X, "predict_proba")
        else:
            log.warning(
                "Ensemble was tuned with a class label predictions metric, "
                "not probabilities. Using weighted mean of class predictions."
            )
            return self._get_weighted_mean_predictions(X, "predict").toarray()


class EnsembleRegressor(Ensemble):
    def _ensemble_validation_score(self, prediction_to_validate=None):
        if prediction_to_validate is None:
            prediction_to_validate = self._averaged_validation_predictions()
        return self._metric.maximizable_score(self._y, prediction_to_validate)

    def predict(self, X):
        return self._get_weighted_mean_predictions(X)


def build_fit_ensemble(
    x,
    y,
    ensemble_size: int,
    timeout: float,
    metric: Metric,
    evaluation_library: EvaluationLibrary,
    encoder: Optional[object] = None,
) -> Ensemble:
    """ Construct an Ensemble of models, optimizing for metric. """
    start_build = time.time()

    log.debug("Building ensemble.")
    if metric.task_type == MetricType.REGRESSION:
        ensemble = EnsembleRegressor(metric, y, evaluation_library)  # type: Ensemble
    elif metric.task_type == MetricType.CLASSIFICATION:
        ensemble = EnsembleClassifier(metric, y, evaluation_library=evaluation_library)
        ensemble._label_encoder = encoder
    else:
        raise ValueError(f"Unknown metric task type {metric.task_type}")

    try:
        # Starting with more models in the ensemble should help against overfitting,
        # but depending on the total ensemble size, it might leave too little room
        # to calibrate the weights or add new models. So we have some adaptive defaults.
        if ensemble_size <= 10:
            ensemble.build_initial_ensemble(1)
        else:
            ensemble.build_initial_ensemble(10)

        remainder = ensemble_size - ensemble._total_model_weights()
        if remainder > 0:
            ensemble.expand_ensemble(remainder)

        build_time = time.time() - start_build
        timeout = timeout - build_time
        log.info(f"Ensemble build took {build_time}s. Fit with timeout {timeout}s.")
        ensemble.fit(x, y, timeout)
    except Exception as e:
        log.warning(f"Error during auto ensemble: {e}", exc_info=True)

    return ensemble
