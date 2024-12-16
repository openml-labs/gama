import copy
from typing import List, Optional, Any, Tuple, Union
import random

import ConfigSpace as cs

from gama.genetic_programming.components import (
    Primitive,
    Terminal,
    PrimitiveNode,
    DATA_TERMINAL,
)
from gama.utilities.config_space import (
    get_estimator_by_name,
    get_hyperparameter_sklearn_name,
    get_internal_output_types,
)


def random_primitive_node(
    output_type: str,
    config_space: cs.ConfigurationSpace,
    exclude: Optional[Primitive] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals."""
    if output_type not in get_internal_output_types():
        raise ValueError(f"Output type {output_type} not supported")

    if exclude is not None:
        temp_config_space = copy.deepcopy(config_space)
        if output_type not in temp_config_space.meta:
            raise ValueError(f"Output type {output_type} not in config_space meta.")
        temp_config_space.add_forbidden_clause(
            cs.ForbiddenEqualsClause(
                temp_config_space.get_hyperparameter(
                    temp_config_space.meta[output_type]
                ),
                exclude.__str__(),
            )
        )
        config = temp_config_space.sample_configuration()
    else:
        config = config_space.sample_configuration()

    if output_type in [DATA_TERMINAL, "preprocessors"]:
        (
            preprocessor_primitive,
            preprocessor_terminals,
        ) = _config_preprocessor_to_primitive_node(
            config, config_space.meta, config_space.get_conditions()
        )
        return PrimitiveNode(
            preprocessor_primitive,
            data_node=DATA_TERMINAL,
            terminals=preprocessor_terminals,
        )
    estimator_primitive, estimator_terminals = _config_estimator_to_primitive_node(
        config, config_space.meta, config_space.get_conditions()
    )
    return PrimitiveNode(
        primitive=estimator_primitive,
        data_node=DATA_TERMINAL,
        terminals=estimator_terminals,
    )


def create_random_expression(
    config_space: cs.ConfigurationSpace,
    min_length: int = 1,
    max_length: int = 3,
) -> PrimitiveNode:
    """Create at least min_length and at most max_length chained PrimitiveNodes."""
    individual_length = random.randint(min_length, max_length)
    return _config_to_primitive_node(
        config=config_space.sample_configuration(),
        config_meta=config_space.meta,
        conditions=config_space.get_conditions(),
        config_length=individual_length,
    )


def extract_valid_hyperparameters(
    cond: cs.conditions, config: cs.Configuration, config_meta: dict, meta_key: str
) -> Union[str, None]:
    """Extract valid hyperparameters from a condition.

    For each supported ConfigSpace condition type, extract the valid hyperparameters
    from the condition. The valid hyperparameters are the hyperparameters that are
    valid for the given condition and configuration based on the meta_key.

    Supported ConfigSpace condition types:
    - EqualsCondition
    - AndConjunction

    Readers are encouraged to add support for more ConfigSpace condition types if
    needed. Open an issue or pull request on the GAMA GitHub repository.

    Parameters
    ----------
    cond : cs.conditions
        A condition of type ConfigSpace.
    config : cs.Configuration
        A configuration of type ConfigSpace.
    config_meta : dict
        The meta information of the ConfigSpace.
    meta_key : str
        The meta key of the ConfigSpace.
    """
    if meta_key not in config_meta:
        raise ValueError(f"Meta key {meta_key} not in config_meta")
    if type(cond) not in [cs.conditions.EqualsCondition, cs.conditions.AndConjunction]:
        raise ValueError(
            f"Condition type {type(cond)} not supported. Refer to "
            f"docstring for supported condition types."
        )
    if isinstance(cond, cs.conditions.EqualsCondition):
        if (
            cond.parent.name == config_meta[meta_key]
            and cond.value == config[config_meta[meta_key]]
        ):
            return cond.child.name
    elif isinstance(cond, cs.conditions.AndConjunction):
        for component in cond.components:
            if (
                component.parent.name == config_meta[meta_key]
                and component.value == config[config_meta[meta_key]]
            ):
                return component.child.name
    return None


def _config_estimator_to_primitive_node(
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    output_type: Optional[str] = None,
) -> Tuple[Primitive, List[Terminal]]:
    """Create a PrimitiveNode from a configuration of type ConfigSpace (estimator).

    Creates a PrimitiveNode from a configuration of type ConfigSpace. Focuses on
    the estimator part of the configuration. It starts by creating a Primitive for
    the selected estimator. Then it determines the valid hyperparameters for the
    estimator based on the conditions. Finally, it creates a Terminal for each valid
    hyperparameter for the estimator.

    Parameters
    ----------
    config : cs.Configuration
        A configuration of type ConfigSpace.
    config_meta : dict
        The meta information of the ConfigSpace.
    conditions : List[Any]
        The conditions of the ConfigSpace.
    output_type : str, optional
        The output type of the PrimitiveNode, by default None.
    """
    if (
        "estimators" not in config_meta
        or config_meta["estimators"] not in config.get_dictionary()
    ):
        raise ValueError(
            f"Configuration {config} does not contain an `estimator` ConfigSpace"
            f"Hyperparameter. Cannot construct estimator PrimitiveNode."
        )
    if output_type is None:
        output_type = "estimators"

    # Create a Primitive for the selected classifier
    estimator_primitive = Primitive(
        identifier=get_estimator_by_name(config[config_meta["estimators"]]),
        output=output_type,
        input=tuple(
            get_hyperparameter_sklearn_name(hp)
            for hp in config
            if hp
            not in [
                config_meta["estimators"],
                config_meta.get("preprocessors"),
            ]
        ),
    )

    # Determine valid hyperparameters for estimators based on conditions
    estimator_valid_hyperparameters = [
        name
        for condition in conditions
        if (
            name := extract_valid_hyperparameters(
                condition, config, config_meta, "estimators"
            )
        )
        is not None
    ]

    # Create a Terminal for each valid hyperparameter for estimators
    estimator_terminals = [
        Terminal(
            identifier=get_hyperparameter_sklearn_name(param),
            value=value,
            output=get_hyperparameter_sklearn_name(param),
            config_space_name=param,
        )
        for param, value in config.items()
        if param in estimator_valid_hyperparameters
        and param
        not in [
            config_meta["estimators"],
            config_meta.get("preprocessors"),
        ]
    ]

    return estimator_primitive, estimator_terminals


def _config_preprocessor_to_primitive_node(
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    output_type: Optional[str] = None,
) -> Tuple[Primitive, List[Terminal]]:
    """Create a PrimitiveNode from a configuration of type ConfigSpace (preprocessor).

    Creates a PrimitiveNode from a configuration of type ConfigSpace. Focuses on
    the preprocessor part of the configuration. It starts by creating a Primitive for
    the selected preprocessor. Then it determines the valid hyperparameters for the
    preprocessor based on the conditions. Finally, it creates a Terminal for each
    valid hyperparameter for the preprocessor.

    Parameters
    ----------
    config : cs.Configuration
        A configuration of type ConfigSpace.
    config_meta : dict
        The meta information of the ConfigSpace.
    conditions : List[Any]
        The conditions of the ConfigSpace.
    output_type : str, optional
        The output type of the PrimitiveNode, by default None.
    """
    if (
        "preprocessors" not in config_meta
        or config_meta["preprocessors"] not in config.get_dictionary()
    ):
        raise ValueError(
            f"Configuration {config} does not contain an `preprocessor` ConfigSpace"
            f"Hyperparameter. Cannot construct preprocessor PrimitiveNode."
        )
    if output_type is None:
        output_type = "preprocessors"

    # Create a Primitive for the selected preprocessor
    preprocessor_primitive = Primitive(
        identifier=get_estimator_by_name(config[config_meta["preprocessors"]]),
        output=output_type,
        input=tuple(
            get_hyperparameter_sklearn_name(hp)
            for hp in config
            if hp not in [config_meta.get("estimators"), config_meta["preprocessors"]]
        ),
    )

    # Determine valid hyperparameters for preprocessor based on conditions
    preprocessor_valid_hyperparameters = [
        name
        for condition in conditions
        if (
            name := extract_valid_hyperparameters(
                condition, config, config_meta, "preprocessors"
            )
        )
        is not None
    ]

    # Create a Terminal for each valid hyperparameter for preprocessor
    preprocessor_terminals = [
        Terminal(
            identifier=get_hyperparameter_sklearn_name(param),
            value=value,
            output=get_hyperparameter_sklearn_name(param),
            config_space_name=param,
        )
        for param, value in config.items()
        if param in preprocessor_valid_hyperparameters
        and param not in [config_meta.get("estimators"), config_meta["preprocessors"]]
    ]

    return preprocessor_primitive, preprocessor_terminals


def _config_to_primitive_node(
    config: cs.Configuration,
    config_meta: dict,
    conditions: List[Any],
    config_length: Optional[int] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode from a configuration. If config_length is specified, the
    PrimitiveNode will have at most config_length PrimitiveNodes."""
    if isinstance(config_length, int) and config_length <= 1:
        estimator_primitive, estimator_terminals = _config_estimator_to_primitive_node(
            config, config_meta, conditions
        )
        return PrimitiveNode(
            estimator_primitive, data_node=DATA_TERMINAL, terminals=estimator_terminals
        )
    estimator_primitive, estimator_terminals = _config_estimator_to_primitive_node(
        config, config_meta, conditions
    )
    (
        preprocessor_primitive,
        preprocessor_terminals,
    ) = _config_preprocessor_to_primitive_node(config, config_meta, conditions)

    # Create a PrimitiveNode for the preprocessor
    preprocessor_node = PrimitiveNode(
        preprocessor_primitive,
        data_node=DATA_TERMINAL,
        terminals=preprocessor_terminals,
    )

    # Create a PrimitiveNode for the classifier and chain it to the preprocessor
    return PrimitiveNode(
        estimator_primitive, data_node=preprocessor_node, terminals=estimator_terminals
    )
