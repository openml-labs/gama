from typing import NamedTuple

DATA_TERMINAL = "data"


class Terminal(NamedTuple):
    """Specifies a specific value for a specific type or input.

    E.g. a value for a hyperparameter for an algorithm.
    """

    value: object
    output: str
    identifier: str

    def __str__(self) -> str:
        """str: e.g. "tol=0.5" """
        return f"{self.output}={format_hyperparameter_value(self.value)}"

    def __repr__(self) -> str:
        """str: e.g. "FastICA.tol=0.5".

        If the hyperparameter is shared across primitives, there is no prefix.
        """
        return f"{self.identifier}={format_hyperparameter_value(self.value)}"


def format_hyperparameter_value(value: object) -> str:
    if isinstance(value, str):
        return f"'{value}'"  # Quoted
    elif callable(value) and hasattr(value, "__name__"):
        return f"{value.__name__}"
    else:
        return str(value)


def find_terminal(primitive_set: dict, terminal_string: str) -> Terminal:
    """Find the Terminal that matches `terminal_string` in `primitive_set`."""
    term_type, _ = terminal_string.split("=")
    for terminal in primitive_set[term_type]:
        if repr(terminal) == terminal_string:
            return terminal
    raise KeyError(f"Could not find Terminal of type '{terminal_string}'.")
