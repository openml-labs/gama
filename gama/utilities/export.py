from sklearn.pipeline import Pipeline

from gama.genetic_programming.components import Individual


def model_to_python(pipeline: Pipeline) -> str:
    """ Generates Python code which sets up the machine learning pipeline represented by `individual`. """
    imports = []
    for name, step in pipeline.steps:
        imports.append(f"from {step.__module__} import {step.__class__.__name__}")

    script = ("from sklearn.pipeline import Pipeline\n" +
              '\n'.join(imports) + '\n\n' +
              'pipeline = ' + str(pipeline) + '\n')

    return script


def individual_to_python(individual: Individual) -> str:
    """ Generates Python code which sets up the machine learning pipeline represented by `individual`. """
    imports = ["from numpy import nan"]  # Used in SimpleImputer
    for name, step in individual.pipeline.steps:
        imports.append(f"from {step.__module__} import {step.__class__.__name__}")

    # The pipeline consists of two steps:
    # - Data Preparation: SimpleImputer and possibly One-hot or Target encoding.
    # - The remainder: additional transformers and an estimator.
    # Because the data preparation step is (currently) defined once and shared across all pipelines,
    # this is not captured in the Individual itself, so we have to extract it from the compiled pipeline.
    steps = []
    n_data_preparation_steps = len(individual.pipeline.steps) - len(individual.primitives)
    for name, step in individual.pipeline.steps[:n_data_preparation_steps]:
        steps.append((name, step))

    for i, primitive_node in enumerate(reversed(individual.primitives)):
        steps.append((str(i), primitive_node.str_nonrecursive))
        for terminal in primitive_node._terminals:
            if callable(terminal.value):
                imports.append(f"from {terminal.value.__module__} import {terminal.value.__name__}")

    steps_str = ',\n'.join([f"('{name}', {step})" for name, step in steps])
    pipeline = f"Pipeline([{steps_str}])"
    script = ("from sklearn.pipeline import Pipeline\n" +
              '\n'.join(imports) + '\n\n' +
              'pipeline = ' + pipeline + '\n')

    return script
