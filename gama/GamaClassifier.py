import inspect
from typing import Union, Optional
import logging

import numpy as np
import pandas as pd
from ConfigSpace import ForbiddenEqualsClause
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import ConfigSpace as cs

from gama.configuration.classification import config_space as clf_config
from gama.data_loading import X_y_from_file
from gama.utilities.metrics import scoring_to_metric, Metric
from .gama import Gama
from .utilities.config_space import get_estimator_by_name

# Avoid stopit from logging warnings every time a pipeline evaluation times out
logging.getLogger("stopit").setLevel(logging.ERROR)
log = logging.getLogger(__name__)


class GamaClassifier(Gama):
    """Gama with adaptations for (multi-class) classification."""

    def __init__(
        self,
        search_space: Optional[cs.ConfigurationSpace] = None,
        scoring: Metric = "neg_log_loss",  # type: ignore
        *args,
        **kwargs,
    ):
        if not search_space:
            # Do this to avoid the whole dictionary being included in the documentation.
            search_space = clf_config

        self._metrics = scoring_to_metric(scoring)

        search_space = self._search_space_check(search_space)

        self._label_encoder = None
        super().__init__(
            *args, search_space=search_space, scoring=scoring, **kwargs
        )  # type: ignore

    def _search_space_check(
        self,
        search_space: cs.ConfigurationSpace,
    ) -> cs.ConfigurationSpace:
        """Check if the search space is valid for classification."""

        # Check if the search space contains a classifier hyperparameter.
        if (
            "estimators" not in search_space.meta
            or (
                search_space.meta["estimators"]
                not in search_space.get_hyperparameters_dict()
            )
            or not isinstance(
                search_space.get_hyperparameter(search_space.meta["estimators"]),
                cs.CategoricalHyperparameter,
            )
        ):
            raise ValueError(
                "The search space must include a hyperparameter for the classifiers "
                "that is a CategoricalHyperparameter with choices for all desired "
                "classifiers. Please double-check the spelling of the name, and review "
                "the `meta` object in the search space configuration located at "
                "`configurations/classification.py`. The `meta` object should contain "
                "a key `estimators` with a value that is the name of the hyperparameter"
                " that contains the classifier choices."
            )

        # Check if the search space contains a preprocessor hyperparameter
        # if it is specified in the meta.
        if (
            "preprocessors" in search_space.meta
            and (
                search_space.meta["preprocessors"]
                not in search_space.get_hyperparameters_dict()
            )
            or "preprocessors" in search_space.meta
            and not isinstance(
                search_space.get_hyperparameter(search_space.meta["preprocessors"]),
                cs.CategoricalHyperparameter,
            )
        ):
            raise ValueError(
                "The search space must include a hyperparameter for the preprocessors "
                "that is a CategoricalHyperparameter with choices for all desired "
                "preprocessors. Please double-check the spelling of the name, and "
                "review the `meta` object in the search space configuration located at "
                "`configurations/classification.py`. The `meta` object should contain "
                "a key `preprocessors` with a value that is the name of the "
                "hyperparameter that contains the preprocessor choices. "
            )

        # Check if the search space contains only classifiers that have predict_proba
        # if the scoring requires probabilities.
        if any(metric.requires_probabilities for metric in self._metrics):
            # we don't want classifiers that do not have `predict_proba`,
            # because then we have to start doing one hot encodings of predictions etc.
            no_proba_clfs = []
            for classifier in search_space.get_hyperparameter(
                search_space.meta["estimators"]
            ).choices:
                estimator = get_estimator_by_name(classifier)
                if (
                    estimator is not None
                    and issubclass(estimator, ClassifierMixin)
                    and not hasattr(estimator(), "predict_proba")
                ):
                    no_proba_clfs.append(classifier)

            log.info(
                f"The following classifiers do not have a predict_proba method "
                f"and will be excluded from the search space: {no_proba_clfs}"
            )
            search_space.add_forbidden_clauses(
                [
                    ForbiddenEqualsClause(
                        search_space.get_hyperparameter(
                            search_space.meta["estimators"]
                        ),
                        classifier,
                    )
                    for classifier in no_proba_clfs
                    if classifier
                ]
            )

        return search_space

    def _predict(self, x: pd.DataFrame):
        """Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array with predictions of shape (N,) where N is len(X).
        """
        y = self.model.predict(x)  # type: ignore
        # Decode the predicted labels - necessary only if ensemble is not used.
        if y[0] not in list(self._label_encoder.classes_):  # type: ignore
            y = self._label_encoder.inverse_transform(y)  # type: ignore
        return y

    def _predict_proba(self, x: pd.DataFrame):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        return self.model.predict_proba(x)  # type: ignore

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]):
        """Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            Data with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is len(x),
             and K is the number of class labels found in `y` of `fit`.
        """
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def predict_proba_from_file(
        self,
        arff_file_path: str,
        target_column: Optional[str] = None,
        encoding: Optional[str] = None,
    ):
        """Predict the class probabilities for input in the arff_file.

        Parameters
        ----------
        arff_file_path: str
            An ARFF file with the same columns as the one that used in fit.
            Target column must be present in file, but its values are ignored.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
            If left None, the last column is taken to be the target.
        encoding: str, optional
            Encoding of the ARFF file.

        Returns
        -------
        numpy.ndarray
            Numpy array with class probabilities.
            The array is of shape (N, K) where N is len(X),
            and K is the number of class labels found in `y` of `fit`.
        """
        x, _ = X_y_from_file(arff_file_path, target_column, encoding)
        x = self._prepare_for_prediction(x)
        return self._predict_proba(x)

    def fit(self, x, y, *args, **kwargs):
        """Should use base class documentation."""
        y_ = y.squeeze() if isinstance(y, pd.DataFrame) else y
        self._label_encoder = LabelEncoder().fit(y_)
        if any(isinstance(yi, str) for yi in y_):
            # If target values are `str` we encode them or scikit-learn will complain.
            y = self._label_encoder.transform(y_)
        self._evaluation_library.determine_sample_indices(stratify=y)

        # Add label information for classification to the scorer such that
        # the cross validator does not encounter unseen labels in smaller
        # data sets during pipeline evaluation.
        for m in self._metrics:
            if "labels" in inspect.signature(m.scorer._score_func).parameters:
                m.scorer._kwargs.update({"labels": y})

        super().fit(x, y, *args, **kwargs)

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)
