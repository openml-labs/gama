from typing import NamedTuple

DATA_TERMINAL = 'data'

# Specifies a specific value for a specific type or input, e.g. a value for a hyperparameter for an algorithm.
Terminal = NamedTuple("Terminal",
                      [("value", object),
                       ("output", str),
                       ("identifier", str)])


def str_format_terminal_value(terminal) -> str:
    if isinstance(terminal.value, str):
        return "'{}'".format(terminal.value)
    elif callable(terminal.value):
        return "{}".format(terminal.value.__name__)
    else:
        return str(terminal.value)


def terminal__str__(terminal) -> str:
    """ e.g. "tol=0.5" """
    return "{}={}".format(terminal.output, str_format_terminal_value(terminal))


def terminal__repr__(terminal) -> str:
    """ e.g. "FastICA.tol=0.5". Note that if the hyperparameter is shared across primitives, there is no prefix. """
    return "{}={}".format(terminal.identifier, str_format_terminal_value(terminal))


Terminal.__str__ = terminal__str__
Terminal.__repr__ = terminal__repr__