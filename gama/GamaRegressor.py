import pandas as pd

from .gama import Gama
from gama.configuration.regression import config_space as reg_config
import ConfigSpace as cs


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

        search_space = self._search_space_check(search_space)

        super().__init__(*args, search_space=search_space, scoring=scoring, **kwargs)

    def _search_space_check(
        self, search_space: cs.ConfigurationSpace
    ) -> cs.ConfigurationSpace:
        """Check if the search space is valid for regression."""

        # Check if the search space contains a regressor hyperparameter.
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
                "The search space must include a hyperparameter for the regressors "
                "that is a CategoricalHyperparameter with choices for all desired "
                "regressors. Please double-check the spelling of the name, and review "
                "the `meta` object in the search space configuration located at "
                "`configurations/regression.py`. The `meta` object should contain "
                "a key `estimators` with a value that is the name of the hyperparameter"
                " that contains the regressor choices."
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
                "`configurations/regression.py`. The `meta` object should contain "
                "a key `preprocessors` with a value that is the name of the "
                "hyperparameter that contains the preprocessor choices. "
            )

        return search_space

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
