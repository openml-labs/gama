import copy
from typing import Tuple, List, Set

from sklearn.base import TransformerMixin
from gama.genetic_programming.components import Individual


def transformers_to_str(transformers: List[TransformerMixin]) -> List[str]:
    """ Format a transformer for code export, removes any mapping. """
    copies = list(map(copy.copy, transformers))
    for transformer in copies:
        if hasattr(transformer, "mapping"):
            transformer.mapping = None  # type: ignore  # ignore no attr 'mapping'
    return list(map(str, copies))


def format_import(o: object) -> str:
    """ Creates the import statement for `o`'s class. """
    if o.__module__.split(".")[-1].startswith("_"):
        module = ".".join(o.__module__.split(".")[:-1])
    else:
        module = o.__module__
    return f"from {module} import {o.__class__.__name__}"


def format_pipeline(steps: List[Tuple[str, str]], name: str = "pipeline"):
    steps_str = ",\n".join([f"('{name}', {step})" for name, step in steps])
    return f"{name} = Pipeline([{steps_str}])\n"


def imports_and_steps_for_individual(
    individual: Individual,
) -> Tuple[Set[str], List[Tuple[str, str]]]:
    """ Determine required imports and steps for the individual's pipeline.

    Returns two lists:
     - one with import statements
     - one with tuples with pipeline step names and step values

    E.g. (["from sklearn.naive_bayes import GaussianNB"], [('0', 'GaussianNB()')])
    """
    imports = ["from numpy import nan", "from sklearn.pipeline import Pipeline"]
    imports += [format_import(step) for name, step in individual.pipeline.steps]

    steps = []
    for i, primitive_node in reversed(list(enumerate(individual.primitives))):
        steps.append((str(i), primitive_node.str_nonrecursive))
        for terminal in primitive_node._terminals:
            if callable(terminal.value) and hasattr(terminal.value, "__name__"):
                imports.append(
                    f"from {terminal.value.__module__} import {terminal.value.__name__}"  # type: ignore # noqa: E501
                )

    return set(imports), steps


def individual_to_python(
    individual: Individual, prepend_steps: List[Tuple[str, TransformerMixin]] = None
) -> str:
    """ Generate code for the machine learning pipeline represented by `individual`. """
    imports, steps = imports_and_steps_for_individual(individual)
    if prepend_steps is not None:
        steps = prepend_steps + steps
        imports = imports.union({format_import(step) for _, step in prepend_steps})
    steps_str = ",\n".join([f"('{name}', {step})" for name, step in steps])
    pipeline = f"Pipeline([{steps_str}])"
    script = "\n".join(sorted(imports)) + "\n\n" + "pipeline = " + pipeline + "\n"

    return script
