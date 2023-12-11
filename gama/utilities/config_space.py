import ConfigSpace as cs
import sklearn
from gama.genetic_programming.components import DATA_TERMINAL
from sklearn.utils import all_estimators


def get_internal_output_types() -> list[str]:
    """Returns the internal ConfigSpace/GAMA output types.

    Returns
    -------
    list[str]
        List of internal ConfigSpace/GAMA output types.
    """
    return [DATA_TERMINAL, "preprocessors", "estimators"]


def get_hyperparameter_sklearn_name(hyperparameter_name: str) -> str:
    """Converts a ConfigSpace hyperparameter name to the name used in sklearn.

    Parameters
    ----------
    hyperparameter_name: str
        Name of the hyperparameter used in ConfigSpace.

    Returns
    -------
    str
        Name of the hyperparameter used in sklearn.
    """
    return hyperparameter_name.split("__")[0]


def get_estimator_by_name(name: str) -> sklearn.base.BaseEstimator:
    """Returns a (sklearn) estimator by name.

    Identify an estimator, which could be a classifier, regressor, or transformer.
    The name should be the same as the estimator's name in sklearn
    (for example, "GaussianNB"). If more than sklearn is supported, on the long term,
    this function could be improved by searching through more than sklearn.

    Parameters
    ----------
    name: str
        Name of the (sklearn) estimator.

    Returns
    -------
    estimator: sklearn.base.BaseEstimator
        The (sklearn) estimator corresponding to the name.
    """
    classifiers = dict(all_estimators(type_filter="classifier"))
    regressors = dict(all_estimators(type_filter="regressor"))
    transformers = dict(all_estimators(type_filter="transformer"))

    all_estimators_dict = classifiers | regressors | transformers

    estimator = all_estimators_dict.get(name)

    if estimator is None:
        raise ValueError(f"Could not find estimator with name {name}.")

    return estimator


def merge_configurations(
    c1: cs.ConfigurationSpace,
    c2: cs.ConfigurationSpace,
    prefix: str = "merged",
    delimiter: str = "_",
) -> cs.ConfigurationSpace:
    """Takes two configuration spaces and merges them together."""
    c1.add_configuration_space(prefix, c2, delimiter)
    return c1
