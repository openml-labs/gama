from sklearn.pipeline import Pipeline

from gama.genetic_programming.own_implementation.components import Individual, PrimitiveNode


def primitive_node_to_sklearn(primitive_node: PrimitiveNode) -> object:
    hyperparameters = {terminal.output: terminal.value for terminal in primitive_node._terminals}
    return primitive_node._primitive._identifier(**hyperparameters)


def compile_individual(individual: Individual, parameter_checks=None, preprocessing_steps=None) -> Pipeline:
    steps = [(str(i), primitive_node_to_sklearn(primitive)) for i, primitive in enumerate(individual.primitives)]
    if preprocessing_steps:
        steps = steps + [(str(i), step) for (i, step) in enumerate(preprocessing_steps, start=len(steps))]
    return Pipeline(list(reversed(steps)))
