import functools
from typing import Any, Optional

import ConfigSpace
from ConfigSpace import Configuration
from gama.genetic_programming.operations import _config_to_primitive_node

from gama.genetic_programming.operator_set import OperatorSet

from gama.genetic_programming.components import Individual
from smac import (
    Scenario,
    AlgorithmConfigurationFacade,
    RandomFacade,
    MultiFidelityFacade,
    HyperparameterOptimizationFacade,
    HyperbandFacade,
    BlackBoxFacade,
)
from smac.facade import AbstractFacade
from smac.initial_design import (
    RandomInitialDesign,
    DefaultInitialDesign,
    FactorialInitialDesign,
    SobolInitialDesign,
    LatinHypercubeInitialDesign,
)

COMPONENTS_MAPPING = {
    "initial_design": {
        "RandomInitialDesign": RandomInitialDesign,
        "DefaultInitialDesign": DefaultInitialDesign,
        "FactorialInitialDesign": FactorialInitialDesign,
        "LatinHypercubeInitialDesign": LatinHypercubeInitialDesign,
        "SobolInitialDesign": SobolInitialDesign,
    },
    "facade": {
        "RandomFacade": RandomFacade,
        "MultiFidelityFacade": MultiFidelityFacade,
        "HyperparameterOptimizationFacade": HyperparameterOptimizationFacade,
        "HyperbandFacade": HyperbandFacade,
        "BlackBoxFacade": BlackBoxFacade,
        "AlgorithmConfigurationFacade": AlgorithmConfigurationFacade,
    },
}


def get_component(
    scenario: Scenario,
    component_type: str,
    component_mapping: dict,
    component_params: dict,
) -> Any:
    """Returns an instance of a component based on its type and parameters."""
    if component_type in component_mapping:
        comp_class = component_mapping[component_type]
    else:
        raise ValueError(
            f"BayesianOptimisation: Invalid component type "
            f"'{component_type}' passed to 'get_component'."
        )
    return comp_class(scenario=scenario, **component_params)


def get_smac(
    configSpace: ConfigSpace.ConfigurationSpace,
    scenario_params: Optional[dict] = None,
    initial_design_params: Optional[dict] = None,
    facade_params: Optional[dict] = None,
) -> AbstractFacade:
    if scenario_params:
        scenario = Scenario(configSpace, **scenario_params)
    else:
        scenario = Scenario(configSpace)

    if initial_design_params:
        if initial_design_params.get("type") is None:
            raise ValueError(
                "BayesianOptimisation: facade_params must contain a 'type' attribute."
            )
        initial_design_params_without_type = initial_design_params.copy()
        initial_design_params_without_type.pop("type")
        initial_design = get_component(
            scenario=scenario,
            component_type=initial_design_params["type"],
            component_mapping=COMPONENTS_MAPPING["initial_design"],
            component_params={**initial_design_params_without_type},
        )
    else:
        initial_design = RandomInitialDesign(scenario=scenario)

    if facade_params:
        if facade_params.get("type") is None:
            raise ValueError(
                "BayesianOptimisation: facade_params must contain a 'type' attribute."
            )
        facade_params_without_type = facade_params.copy()
        facade_params_without_type.pop("type")
        facade = get_component(
            scenario=scenario,
            component_type=facade_params["type"],
            component_mapping=COMPONENTS_MAPPING["facade"],
            component_params={
                **facade_params_without_type,
                "initial_design": initial_design,
                "target_function": dummy_smac_train,
            },
        )
    else:
        facade = AlgorithmConfigurationFacade(
            scenario=scenario,
            initial_design=initial_design,
            target_function=dummy_smac_train,
        )

    return facade


def dummy_smac_train(config: Configuration, seed: int = 0) -> None:
    _, _ = config, seed
    raise NotImplementedError(
        "BayesianOptimisation: dummy_smac_train should not have been called."
        "Current version does not support smac.Optimize() training procedure in "
        "favour to the Ask&Tell SMAC's interface. Operations.evaluate should be used "
        "instead through the GAMA multi-processing interface."
    )


def config_to_individual(
    config: Configuration,
    operations: OperatorSet,
) -> Individual:
    """Convert a SMAC configuration to a GAMA individual."""
    if config is None:
        raise ValueError(
            "BayesianOptimisation: config_to_individual received a None config."
        )
    if (config_space := operations.get_search_space()) is None:  # type: ignore
        raise ValueError(
            "BayesianOptimisation: Operations.get_search_space() returned None."
        )
    if (
        "preprocessors" in config_space.meta
        and config_space.meta["preprocessors"] in config.keys()
    ):
        max_length = 2
    else:
        max_length = 1
    return Individual(
        main_node=_config_to_primitive_node(
            config=config,
            config_meta=config_space.meta,
            conditions=config_space.get_conditions(),
            config_length=max_length,
        ),
        to_pipeline=operations._safe_compile or operations._compile,
    )


def validate_info_config(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if not hasattr(result, "config") or result.config is None:
            raise ValueError(
                f"BayesianOptimisation: Function "
                f"'{func.__name__}' returned info with None config."
            )
        return result

    return wrapper


def validate_future_valid(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        future = args[0]
        if future is None or not hasattr(future, "result") or future.result is None:
            raise ValueError(
                f"BayesianOptimisation: Function '{func.__name__}' "
                f"received a Future object or its result is not valid."
            )

        result = future.result
        if not hasattr(result, "individual") or result.individual is None:
            raise ValueError(
                "BayesianOptimisation: Future object's result is missing 'individual' "
                "attribute."
            )

        if (
            not hasattr(result.individual, "fitness")
            or result.individual.fitness is None
        ):
            raise ValueError(
                "BayesianOptimisation: Future object's result's individual is missing "
                "'fitness' attribute."
            )

        required_attrs = ["values", "wallclock_time", "start_time", "process_time"]

        for attr in required_attrs:
            if not hasattr(result.individual.fitness, attr):
                raise ValueError(
                    f"BayesianOptimisation: Future object's result's individual's "
                    f"fitness is missing '{attr}' attribute."
                )

        if hasattr(result.individual.fitness, "start_time") and not callable(
            getattr(result.individual.fitness.start_time, "timestamp", None)
        ):
            raise ValueError(
                "BayesianOptimisation: start_time object does not have a timestamp "
                "method."
            )

        return func(*args, **kwargs)

    return wrapper
