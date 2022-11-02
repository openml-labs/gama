from typing import Union, Optional, List
import numpy as np
import pandas as pd
from .gama import Gama
from gama.data_loading import file_to_pandas, X_y_from_file
from gama.genetic_programming.components import Individual
from gama.configuration.clustering import cluster_config
from gama.utilities.metrics import scoring_to_metric


class GamaCluster(Gama):

    """ Gama with adaptations for clustering. """

    def __init__(self, config=None, scoring="calinski_harabasz", *args, **kwargs):

        if not config:
            # Do this to avoid the whole dictionary being included in the documentation.
            config = cluster_config
        self._metrics = scoring_to_metric(scoring)
        self._label_encoder = None

        super().__init__(*args, **kwargs, config=config, scoring=scoring)

    def fit(self, x, y=None, *args, **kwargs):

        if y is not None:
            y = y.squeeze() if isinstance(y, pd.DataFrame) else y
            self._evaluation_library.determine_sample_indices(stratify=y)

        super().fit(x, y, *args, **kwargs)

    def _predict(self, x: pd.DataFrame):
        """ Predict the cluster for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe the same number of columns as that of X of `fit`.

        Returns
        -------
        numpy.ndarray
            Array with cluster predictions of shape (N,) where N is len(X).
        """

        self.assigned_labels = self.model.steps[-1][1].labels_
        # self.assigned_labels = self.model.fit_predict(x)
        return self.assigned_labels

    def score(
            self,
            x: Union[pd.DataFrame, np.ndarray],
            y: Optional[pd.Series] = None
    ) -> float:

        """ Calculate `self.scoring` metric of the model on (x, y).

        Parameters
        ----------
        x: pandas.DataFrame or numpy.ndarray
            Data to predict target values for.
        y: pandas.Series or numpy.ndarray
            True values for the target.

        Returns
        -------
        float
            The score obtained on the given test data according to the `scoring` metric.
        """

        # TODO: raise error (default=ch, if y is provided then user has to define scoring)
        if y is not None:
            return self._metrics[0].score(y, self.assigned_labels)
        else:
            return self._metrics[0].score(x, self.assigned_labels)

    def fit_from_file(
            self,
            file_path: str,
            target_column: Optional[str] = None,
            encoding: Optional[str] = None,
            warm_start: Optional[List[Individual]] = None,
            **kwargs,
    ) -> None:

        """ Find and fit a model to predict the target column (last) from other columns.

        Parameters
        ----------
        file_path: str
            Path to a csv or ARFF file containing the training data.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
        encoding: str, optional
            Encoding of the file.
        warm_start: List[Individual], optional (default=None)
            A list of individual to start the search  procedure with.
            If None is given, random start candidates are generated.
        **kwargs:
            Any additional arguments for calls to pandas.read_csv or arff.load.

        """
        if target_column is None:
            x = file_to_pandas(file_path, encoding, **kwargs)
            self.fit(x, target_column, warm_start)
        else:
            x, y = X_y_from_file(file_path, target_column, encoding, **kwargs)
            self.fit(x, y, warm_start)

    def predict_from_file(
            self,
            file_path: str,
            target_column: Optional[str] = None,
            encoding: Optional[str] = None,
            **kwargs,
    ) -> np.ndarray:

        """ Predict the cluster for input found in the ARFF file.

        Parameters
        ----------
        file_path: str
            A csv or ARFF file with the same columns as the one that used in fit.
            Target column must be present in file, but its values are ignored.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
        encoding: str, optional
            Encoding of the ARFF file.
        **kwargs:
            Any additional arguments for calls to pandas.read_csv or arff.load.

        Returns
        -------
        numpy.ndarray
            array with predictions for each row in the ARFF file.
        """
        if target_column is None:
            x = file_to_pandas(file_path, encoding, **kwargs)
            return self.predict(x)

        else:
            x, _ = X_y_from_file(file_path, split_column=target_column, encoding=encoding, **kwargs)
            return self.predict(x)

    def score_from_file(
            self,
            file_path: str,
            target_column: Optional[str] = None,
            encoding: Optional[str] = None,
            **kwargs,
    ) -> float:

        """ Calculate `self.scoring` metric of the model on data in the file.

        Parameters
        ----------
        file_path: str
            A csv or ARFF file with which to calculate the score.
        target_column: str, optional (default=None)
            Specifies which column the model should predict.
        encoding: str, optional
            Encoding of the ARFF file.
        **kwargs:
            Any additional arguments for calls to pandas.read_csv or arff.load.

        Returns
        -------
        float
            The score obtained on the given test data according to the `scoring` metric.
        """
        if target_column is None:
            x = file_to_pandas(file_path, encoding, **kwargs)
            return self.score(x, target_column)

        else:
            x, y = X_y_from_file(file_path, split_column=target_column, encoding=encoding, **kwargs)
            return self.score(x, y)
