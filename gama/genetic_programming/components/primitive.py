from typing import Callable, NamedTuple, Tuple


class Primitive(NamedTuple):
    """ Defines an operator which takes input and produces output.

    E.g. a preprocessing or classification algorithm.
    """

    input: Tuple[str]
    output: str
    identifier: Callable

    def __str__(self):
        """ str: e.g. "FastICA" """
        return self.identifier.__name__

    def __repr__(self):
        return str(self)
