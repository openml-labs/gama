from typing import Tuple, List

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from gama.genetic_programming.components import Individual


def model_to_python(pipeline: Pipeline) -> str:
    """ Generate code for the machine learning pipeline represented by `individual`. """
    imports = []
    for name, step in pipeline.steps:
        imports.append(f"from {step.__module__} import {step.__class__.__name__}")

    script = (
        "from sklearn.pipeline import Pipeline\n"
        + "\n".join(imports)
        + "\n\n"
        + "pipeline = "
        + str(pipeline)
        + "\n"
    )

    return script


def imports_and_steps_for_individual(
    individual: Individual,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """ Determine required imports and steps for the individual's pipeline.

    Returns two lists:
     - one with import statements
     - one with tuples with pipeline step names and step values

    E.g. (["from sklearn.naive_bayes import GaussianNB"], [('0', 'GaussianNB()')])
    """
    imports = ["from numpy import nan", "from sklearn.pipeline import Pipeline"]
    for name, step in individual.pipeline.steps:
        # sklearn often contains classes in 'private' submodules
        if step.__module__.split(".")[-1].startswith("_"):
            module = ".".join(step.__module__.split(".")[:-1])
        else:
            module = step.__module__
        imports.append(f"from {module} import {step.__class__.__name__}")

    # The pipeline consists of two steps:
    # - Data Preparation: SimpleImputer and possibly One-hot or Target encoding.
    # - The remainder: additional transformers and an estimator.
    # Because the data preparation step is (currently) defined once
    # and shared across all pipelines, this is not captured in the Individual itself.
    # We have to extract it from the compiled pipeline.
    steps = []
    n_data_preparation_steps = len(individual.pipeline.steps) - len(
        individual.primitives
    )
    for name, step in individual.pipeline.steps[:n_data_preparation_steps]:
        steps.append((name, step))

    for i, primitive_node in reversed(list(enumerate(individual.primitives))):
        steps.append((str(i), primitive_node.str_nonrecursive))
        for terminal in primitive_node._terminals:
            if callable(terminal.value) and hasattr(terminal.value, "__name__"):
                imports.append(
                    f"from {terminal.value.__module__} import {terminal.value.__name__}"  # type: ignore # noqa: E501
                )

    return imports, steps


def individual_to_python(
    individual: Individual, prepend_steps: List[Tuple[str, TransformerMixin]] = None
) -> str:
    """ Generate code for the machine learning pipeline represented by `individual`. """
    imports, steps = imports_and_steps_for_individual(individual)
    if prepend_steps is not None:
        steps = prepend_steps + steps
        imports = [
            f"from {step.__module__} import {step.__class__.__name__}"
            for name, step in prepend_steps
        ] + imports
    steps_str = ",\n".join([f"('{name}', {step})" for name, step in steps])
    pipeline = f"Pipeline([{steps_str}])"
    script = "\n".join(sorted(imports)) + "\n\n" + "pipeline = " + pipeline + "\n"

    return script
