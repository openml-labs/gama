from typing import NamedTuple

DATA_TERMINAL = "data"


class Terminal(NamedTuple):
    """ Specifies a specific value for a specific type or input.

    E.g. a value for a hyperparameter for an algorithm.
    """

    value: object
    output: str
    identifier: str

    def __str__(self):
        """ str: e.g. "tol=0.5" """
        return f"{self.output}={format_hyperparameter_value(self.value)}"

    def __repr__(self):
        """ str: e.g. "FastICA.tol=0.5".

        If the hyperparameter is shared across primitives, there is no prefix.
        """
        return f"{self.identifier}={format_hyperparameter_value(self.value)}"


def format_hyperparameter_value(value: object) -> str:
    if isinstance(value, str):
        return f"'{value}'"  # Quoted
    elif callable(value) and hasattr(value, "__name__"):
        return f"{value.__name__}"  # type: ignore
    else:
        return str(value)
