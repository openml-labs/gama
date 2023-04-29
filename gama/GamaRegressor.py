import pandas as pd

from .gama import Gama
from gama.configuration.regression import reg_config


class GamaRegressor(Gama):
    """Gama with adaptations for regression."""

    def __init__(
        self, search_space=None, scoring="neg_mean_squared_error", *args, **kwargs
    ):
        """ """
        # Empty docstring overwrites base __init__ doc string.
        # Prevents duplication of the __init__ doc string on the API page.

        if not search_space:
            search_space = reg_config
        super().__init__(*args, search_space=search_space, scoring=scoring, **kwargs)

    def _predict(self, x: pd.DataFrame):
        """Predict the target for input X.

        Parameters
        ----------
        x: pandas.DataFrame
            A dataframe the same number of columns as that of X of `fit`.

        Returns
        -------
        numpy.ndarray
            Array with predictions of shape (N,) where N is len(X).
        """
        return self.model.predict(x)  # type: ignore
