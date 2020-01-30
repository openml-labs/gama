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

    shared_hyperparameter_types = {}
    # We have to make sure the str-keys are evaluated first: they describe shared hyperparameters
    # We can not rely on order-preserving dictionaries as this is not in the Python 3.5 specification.
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
                    # This allows users to define illegal hyperparameter combinations, but is not a terminal.
                    parameter_checks[key.__name__] = param_values[0]
                else:
                    hyperparameter_types.append(key.__name__ + '.' + name)
                    for value in param_values:
                        pset[key.__name__ + '.' + name].append(
                            Terminal(value=value, output=name, identifier=key.__name__ + '.' + name))

            # After registering the hyperparameter types, we can register the operator itself.
            if hasattr(key, 'pclass'):
                transformer_tags = ["DATA_PREPROCESSING", "FEATURE_SELECTION", "DATA_TRANSFORMATION", "FEATURE_EXTRACTION"]
                predictor_tags = ["CLASSIFICATION", "REGRESSION", "TIME_SERIES_FORECASTING"]
                family = key.pclass.metadata.query()['primitive_family']
                if family in transformer_tags:
                    pset[DATA_TERMINAL].append(Primitive(input=hyperparameter_types, output=DATA_TERMINAL, identifier=key))
                elif family in predictor_tags:
                    pset["prediction"].append(Primitive(input=hyperparameter_types, output="prediction", identifier=key))
                else:
                    raise TypeError("{} has unknown family '{}'".format(key, family))
            else:
                if issubclass(key, sklearn.base.TransformerMixin):
                    pset[DATA_TERMINAL].append(Primitive(input=hyperparameter_types, output=DATA_TERMINAL, identifier=key))
                elif issubclass(key, sklearn.base.ClassifierMixin) or issubclass(key, sklearn.base.RegressorMixin):
                    pset["prediction"].append(Primitive(input=hyperparameter_types, output="prediction", identifier=key))
                else:
                    raise TypeError("Expected {} to be either subclass of "
                                    "TransformerMixin, RegressorMixin or ClassifierMixin.".format(key))
        else:
            raise TypeError('Encountered unknown type as key in dictionary.'
                            'Keys in the configuration should be str or class.')

    return pset, parameter_checks


def merge_configurations(c1, c2):
    """ Takes two configurations and merges them together. """
    # Should refactor out 6 indentation levels
    merged: Dict[Any, Any] = defaultdict(lambda: None, c1)
    for algorithm, hyperparameters2 in c2.items():
        if algorithm not in merged:
            merged[algorithm] = hyperparameters2
        else:
            hyperparameters1 = merged[algorithm]
            if isinstance(hyperparameters1, list) and isinstance(hyperparameters2, list):
                #  they hyperparameters shared across algorithms
                merged[algorithm] = list(set(hyperparameters1 + hyperparameters2))
            else:
                for hyperparameter, values in hyperparameters2.items():
                    if hyperparameter not in hyperparameters1:
                        hyperparameters1[hyperparameter] = values
                    else:
                        values1 = hyperparameters1[hyperparameter]
                        if isinstance(values1, dict) and isinstance(values, dict):
                            hyperparameters1[hyperparameter] = {**values1, **values}
                        elif isinstance(values1, type(values)):
                            hyperparameters1[hyperparameter] = list(set(list(values1) + list(values)))
                        else:
                            raise TypeError(f'Could not merge values of {algorithm}.{hyperparameter}:'
                                            f'{hyperparameters1} vs. {hyperparameters2}')
    return merged
