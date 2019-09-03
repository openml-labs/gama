from typing import Callable, NamedTuple, Tuple

# Defines an operator which takes input and produces output, e.g. a preprocessing or classification algorithm.
Primitive = NamedTuple("Primitive",
                       [("input", Tuple[str]),
                        ("output", str),
                        ("identifier", Callable)])


def primitive__str__(primitive) -> str:
    """ str: e.g. "FastICA" """
    return primitive.identifier.__name__


Primitive.__str__ = primitive__str__
Primitive.__repr__ = primitive__str__
