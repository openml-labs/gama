from collections import defaultdict
from typing import Dict, Any

import sklearn

from gama.genetic_programming.components import Primitive, Terminal, DATA_TERMINAL


def pset_from_config(configuration):
    """ Create a pset for the given configuration dictionary.

    Given a configuration dictionary specifying operators (e.g. sklearn
    estimators), their hyperparameters and values for each hyperparameter,
    create a gp.PrimitiveSetTyped that contains:

        - For each operator a primitive
        - For each possible hyperparameter-value combination a unique terminal

    Side effect: Imports the classes of each primitive.

    Returns the given Pset.
    """

    pset = defaultdict(list)
    parameter_checks = {}

    # Make sure the str-keys are evaluated first, they describe shared hyperparameters.
    # Order-preserving dictionaries are not in the Python 3.6 specification.
    sorted_keys = reversed(sorted(configuration.keys(), key=lambda x: str(type(x))))
    for key in sorted_keys:
        values = configuration[key]
        if isinstance(key, str):
            # Specification of shared hyperparameters
            for value in values:
                pset[key].append(Terminal(value=value, output=key, identifier=key))
        elif isinstance(key, object):
            # Specification of operator (learner, preprocessor)
            hyperparameter_types = []
            for name, param_values in sorted(values.items()):
                # We construct a new type for each hyperparameter, so we can specify
                # it as terminal type, making sure it matches with expected
                # input of the operators. Moreover it automatically makes sure that
                # crossover only happens between same hyperparameters.
                if isinstance(param_values, list) and not param_values:
                    # An empty list indicates a shared hyperparameter
                    hyperparameter_types.append(name)
                elif name == "param_check":
                    # This allows users to define illegal hyperparameter combinations,
                    # but is not a terminal.
                    parameter_checks[key.__name__] = param_values[0]
                else:
                    hp_name = f"{key.__name__}.{name}"
                    hyperparameter_types.append(hp_name)
                    for value in param_values:
                        pset[hp_name].append(
                            Terminal(value=value, output=name, identifier=hp_name,)
                        )

            # After registering the hyperparameter types,
            # we can register the operator itself.
            transformer_tags = [
                "DATA_PREPROCESSING",
                "FEATURE_SELECTION",
                "DATA_TRANSFORMATION",
            ]
            if issubclass(key, sklearn.base.TransformerMixin) or (
                hasattr(key, "metadata")
                and key.metadata.query()["primitive_family"] in transformer_tags
            ):
                pset[DATA_TERMINAL].append(
                    Primitive(
                        input=hyperparameter_types, output=DATA_TERMINAL, identifier=key
                    )
                )
            elif issubclass(key, sklearn.base.ClassifierMixin) or (
                hasattr(key, "metadata")
                and key.metadata.query()["primitive_family"] == "CLASSIFICATION"
            ):
                pset["prediction"].append(
                    Primitive(
                        input=hyperparameter_types, output="prediction", identifier=key
                    )
                )
            elif issubclass(key, sklearn.base.RegressorMixin) or (
                hasattr(key, "metadata")
                and key.metadata.query()["primitive_family"] == "REGRESSION"
            ):
                pset["prediction"].append(
                    Primitive(
                        input=hyperparameter_types, output="prediction", identifier=key
                    )
                )
            else:
                raise TypeError(
                    f"Expected {key} to be either subclass of "
                    "TransformerMixin, RegressorMixin or ClassifierMixin."
                )
        else:
            raise TypeError(
                "Encountered unknown type as key in dictionary."
                "Keys in the configuration should be str or class."
            )

    return pset, parameter_checks


def merge_configurations(c1, c2):
    """ Takes two configurations and merges them together. """
    # Should refactor out 6 indentation levels
    merged: Dict[Any, Any] = defaultdict(lambda: None, c1)
    for algorithm, hparams2 in c2.items():
        if algorithm not in merged:
            merged[algorithm] = hparams2
            continue

        hparams = merged[algorithm]
        if isinstance(hparams, list) and isinstance(hparams2, list):
            merged[algorithm] = list(set(hparams + hparams2))
            continue  # Here the algorithm is actually a shared hyperparameter.

        for hyperparameter, values in hparams2.items():
            if hyperparameter not in hparams:
                hparams[hyperparameter] = values
                continue  # Hyperparameter only specified in one configuration.

            values1 = hparams[hyperparameter]
            if isinstance(values1, dict) and isinstance(values, dict):
                hparams[hyperparameter] = {**values1, **values}
            elif isinstance(values1, type(values)):
                # Both are ranges, arrays or lists.
                hparams[hyperparameter] = list(set(list(values1) + list(values)))
            else:
                raise TypeError(
                    f"Could not merge values of {algorithm}.{hyperparameter}:"
                    f"{hparams} vs. {hparams2}"
                )
    return merged
