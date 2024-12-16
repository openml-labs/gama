from typing import NamedTuple

DATA_TERMINAL = "data"


class Terminal(NamedTuple):
    """Specifies a specific value for a specific type or input.

    E.g. a value for a hyperparameter for an algorithm.

    It is important to note that you should use the hyperparameter's sklearn name as
    your output and identifier. If your name contains `__estimatorName`, you should
    remove it (e.g. by using string split). More information may be found in the
    documentation for the `get_hyperparameter_sklearn_name` function.

    Furthermore, the `config_space_name` is the name of the Config Space's
    hyperparameter. As a result, this is the name formed by the `__estimatorName` and
    the name of the hyperparameter.
    """

    value: object
    output: str
    identifier: str
    config_space_name: str = "Not Specified"

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
