from collections import defaultdict
import itertools
from typing import Dict, Any, Optional, Union, List, Callable, Tuple

import sklearn

from gama.genetic_programming.components import Primitive, Terminal


def pset_from_config(
    configuration: Dict[Union[str, object], Any]
) -> Tuple[Dict[str, List], Dict[str, Callable]]:
    """Create a pset for the given configuration dictionary.

    Given a configuration dictionary specifying operators (e.g. sklearn
    estimators), their hyperparameters and values for each hyperparameter,
    create a gp.PrimitiveSetTyped that contains:

        - For each operator a primitive
        - For each possible hyperparameter-value combination a unique terminal

    Side effect: Imports the classes of each primitive.

    returns:
        pset - Dict[str, List]:
            maps return-types to a list of Primitives and/or Terminals
        parameter_check - Dict[str, Callable]:
            maps Primitive name to a check for the validity of the hp configuration
    """

    pset: Dict[str, List[Union[Primitive, Terminal]]] = defaultdict(list)
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
        elif isinstance(key, type):
            # Specification of operator (learner, preprocessor)
            hyperparameter_types: List[str] = []
            data_input = values.get("_input", "numeric_data")
            hyperparameter_types.append(data_input)
            for name, param_values in sorted(values.items()):
                # We construct a new type for each hyperparameter, so we can specify
                # it as terminal type, making sure it matches with expected
                # input of the operators. Moreover it automatically makes sure that
                # crossover only happens between same hyperparameters.
                if isinstance(param_values, list) and not param_values:
                    # An empty list indicates a shared hyperparameter
                    hyperparameter_types.append(name)
                elif isinstance(param_values, dict) \
                        and all([isinstance(_k, type) for _k in param_values.keys()]):
                    for sub_key, sub_hyperparameters in param_values.items():
                        sub_hps_for_encoder = []

                        for enc_param, sub_hyperparams in sub_hyperparameters.items():
                            hp_name = f"{key.__name__}.{name}.{sub_key.__name__}.{enc_param}"
                            sub_hps_for_encoder.append(hp_name)
                            for sub_param_value in sub_hyperparams:
                                pset[hp_name].append(
                                    Terminal(
                                        value=sub_param_value,
                                        output=enc_param,
                                        identifier=hp_name,
                                    )
                                )
                        hp_name = f"{key.__name__}.{name}"
                        if hp_name not in hyperparameter_types:
                            hyperparameter_types.append(hp_name)
                        pset[hp_name].append(
                            Primitive(
                                input=tuple(sub_hps_for_encoder),
                                output=name,
                                identifier=sub_key,
                                data_input="dont_remove",
                            )
                        )
                elif name == "param_check":
                    # This allows users to define illegal hyperparameter combinations,
                    # but is not a terminal.
                    parameter_checks[key.__name__] = param_values[0]
                elif not name.startswith("_"):
                    hp_name = f"{key.__name__}.{name}"
                    hyperparameter_types.append(hp_name)
                    for value in param_values:
                        pset[hp_name].append(
                            Terminal(
                                value=value,
                                output=name,
                                identifier=hp_name,
                            )
                        )

            # After registering the hyperparameter types,
            # we can register the operator itself.
            if issubclass(key, sklearn.base.TransformerMixin):
                output = values.get("_output", "numeric_data")
                pset[output].append(
                    Primitive(
                        input=tuple(hyperparameter_types),
                        output=output,
                        identifier=key,
                        data_input=data_input,
                    )
                )
            elif issubclass(key, sklearn.base.ClassifierMixin):
                output = values.get("_output", "prediction")
                pset[output].append(
                    Primitive(
                        input=tuple(hyperparameter_types),
                        output=output,
                        identifier=key,
                        data_input=data_input,
                    )
                )
            elif issubclass(key, sklearn.base.RegressorMixin):
                output = values.get("_output", "prediction")
                pset[output].append(
                    Primitive(
                        input=tuple(hyperparameter_types),
                        output=output,
                        identifier=key,
                        data_input=data_input,
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


def merge_configurations(c1: Dict, c2: Dict) -> Dict:
    """Takes two configurations and merges them together."""
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


def compute_reachability(
    pset: Dict, pipeline_input: Optional[str] = None
) -> Dict[str, int]:
    """Calculates the minimum number of primitives required to reach data types."""
    if pipeline_input is None:
        pipeline_input = next(
            k
            for k in pset
            if k.endswith("data") and any(isinstance(t, Terminal) for t in pset[k])
        )

    reachability = {pipeline_input: 0}
    reachability_updated = True
    while reachability_updated:
        reachability_updated = False
        for item in [
            p for p in itertools.chain(*pset.values()) if isinstance(p, Primitive)
        ]:
            reachable = item.data_input in reachability
            new_return_type = item.output not in reachability
            if reachable and (
                new_return_type
                or reachability[item.output] > reachability[item.output] + 1
            ):
                reachability[item.output] = reachability[item.data_input] + 1
                reachability_updated = True
    return reachability


def compute_minimal_pipeline_length(
    pset: Dict, pipeline_input: str = "data", pipeline_output: str = "prediction"
) -> int:
    """Calculates the minimum number of primitives required to reach data types."""
    reachability = compute_reachability(pset, pipeline_input)
    return reachability[pipeline_output]


def remove_primitives_with_unreachable_input(
    pset: Dict, pipeline_input: str = "data"
) -> Dict:
    reachability = compute_reachability(pset, pipeline_input)
    for return_type, prims_and_terms in pset.items():
        for pt in prims_and_terms:
            if isinstance(pt, Primitive) and pt.data_input not in reachability:
                print(pt)
    return {
        return_type: [
            pt
            for pt in prims_and_terms
            if not (isinstance(pt, Primitive) and pt.data_input not in reachability and pt.data_input != "dont_remove")
        ]
        for return_type, prims_and_terms in pset.items()
    }
