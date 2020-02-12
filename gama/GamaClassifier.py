import inspect
from typing import Union

import numpy as np
import pandas as pd
from d3m.container import DataFrame as D3MDataFrame
from d3m.metadata import base as mdbase
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from .gama import Gama
from gama.data import X_y_from_arff
from gama.configuration.classification import clf_config
from gama.utilities.metrics import scoring_to_metric


class GamaClassifier(Gama):
    """ Wrapper for the toolbox logic executing the AutoML pipeline for (multi-class) classification. """
    def __init__(self, config=None, scoring='neg_log_loss', *args, **kwargs):
        if not config:
            # Do this to avoid the whole dictionary being included in the documentation.
            config = clf_config

        self._metrics = scoring_to_metric(scoring)
        if any(metric.requires_probabilities for metric in self._metrics):
            # we don't want classifiers that do not have `predict_proba`, because then we have to
            # start doing one hot encodings of predictions etc.
            config = {alg: hp for (alg, hp) in config.items()
                      if not (inspect.isclass(alg) and issubclass(alg, ClassifierMixin)and not hasattr(alg(), 'predict_proba'))}

        self._label_encoder = None
        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def _predict(self, x: pd.DataFrame):
        """ Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            array with predictions of shape (N,) where N is the length of the first dimension of X.
        """
        y = self.model.predict(x)
        # Decode the predicted labels - necessary only if ensemble is not used.
        if self._label_encoder is not None and y[0] not in self._label_encoder.classes_:
            y = self._label_encoder.inverse_transform(y)
        return y

    def _predict_proba(self, x: pd.DataFrame):
        """ Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is the length of the
            first dimension of x, and K is the number of class labels found in `y` of `fit`.
        """
        return self.model.predict_proba(x)

    def predict_proba(self, x: Union[pd.DataFrame, np.ndarray]):
        """ Predict the class probabilities for input x.

        Predict target for x, using the best found pipeline(s) during the `fit` call.

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            A dataframe or numpy array with the same number of columns as the input to `fit`.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, K) with class probabilities where N is the length of the
            first dimension of x, and K is the number of class labels found in `y` of `fit`.
        """
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
            for col in self._X.columns:
                x[col] = x[col].astype(self._X[col].dtype)
        return self._predict_proba(x)

    def predict_proba_arff(self, arff_file_path: str):
        """ Predict the class probabilities for input in the arff_file, must have empty target column.

        Predict target for X, using the best found pipeline(s) during the `fit` call.

        :param arff_file_path: str

        :return: a numpy array with class probabilities. The array is of shape (N, K) where N is the length of the
            first dimension of X, and K is the number of class labels found in `y` of `fit`.
        """
        X, _ = X_y_from_arff(arff_file_path)
        return self._predict_proba(X)

    def fit(self, x, y, *args, **kwargs):
        """ Should use base class documentation. """
        # If a D3M data frame, not the semantic types, so we can restore them below
        if isinstance(y, D3MDataFrame):
            y_ = y.squeeze()
            col_semantic_types = y.metadata.query_column(0)['semantic_types']
        elif isinstance(y, pd.DataFrame):
            y_ = y.squeeze()
            col_semantic_types = None
        else:
            y_ = y
            col_semantic_types = None
        self._label_encoder = LabelEncoder().fit(y_)
        if any([isinstance(yi, str) for yi in y_]):
            # If target values are `str` we encode them or scikit-learn will complain.
            y = self._label_encoder.transform(y_)
            # D3M categorical predictors barf if metadata is absent
            y = D3MDataFrame(y, generate_metadata=True)
            if col_semantic_types is not None:
                for stype in col_semantic_types:
                    y.metadata = y.metadata.add_semantic_type((mdbase.ALL_ELEMENTS, 0), stype)
        super().fit(x, y, *args, **kwargs)

    def _encode_labels(self, y):
        self._label_encoder = LabelEncoder().fit(y)
        return self._label_encoder.transform(y)
