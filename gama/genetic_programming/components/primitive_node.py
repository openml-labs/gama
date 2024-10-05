import ast
from typing import List, Union, cast


from .terminal import DATA_TERMINAL, Terminal
from .primitive import Primitive
import ConfigSpace as cs

from ...utilities.config_space import (
    get_hyperparameter_sklearn_name,
    get_estimator_by_name,
)


class PrimitiveNode:
    """An instantiation for a Primitive with specific Terminals.

    Parameters
    ----------
    primitive: Primitive
        The Primitive type of this PrimitiveNode.
    data_node: PrimitiveNode
        The PrimitiveNode that specifies all preprocessing before this PrimitiveNode.
    terminals: List[Terminal]
        A list of terminals matching the `primitive`.
    """

    def __init__(
        self,
        primitive: Primitive,
        data_node: Union["PrimitiveNode", str],
        terminals: List[Terminal],
    ):
        self._primitive = primitive
        self._data_node = data_node
        self._terminals = sorted(terminals, key=lambda t: str(t))

    def __str__(self) -> str:
        """Recursively stringify all primitive nodes (primitive and hyperparameters).

        Examples: - "GaussianNB(data)"
                  - "BernoulliNB(data, alpha=1.0)"
                  - "BernoulliNB(FastICA(data, tol=0.5), alpha=1.0)"
        """
        if not self._terminals:
            return f"{self._primitive}({self._data_node})"
        terminal_str = ", ".join([repr(terminal) for terminal in self._terminals])
        return f"{self._primitive}({self._data_node}, {terminal_str})"

    @property
    def str_nonrecursive(self) -> str:
        """Stringify primitive node with its hyperparameter configuration

        Examples: - "GaussianNB()"
                  - "BernoulliNB(alpha=1.0)"
        """
        terminal_str = ", ".join([str(terminal) for terminal in self._terminals])
        return f"{self._primitive}({terminal_str})"

    def copy(self) -> "PrimitiveNode":
        """Copies the object. Shallow for terminals, deep for data_node."""
        if isinstance(self._data_node, str) and self._data_node == DATA_TERMINAL:
            data_node_copy = DATA_TERMINAL  # type: Union[str, PrimitiveNode]
        elif isinstance(self._data_node, PrimitiveNode):
            data_node_copy = self._data_node.copy()
        return PrimitiveNode(
            primitive=self._primitive,
            data_node=data_node_copy,
            terminals=self._terminals.copy(),
        )

    @classmethod
    def from_string(
        cls, string: str, config_space: cs.ConfigurationSpace, strict: bool = True
    ) -> "PrimitiveNode":
        """Create a PrimitiveNode from string formatted like PrimitiveNode.__str__

        Parameters
        ----------
        string: str
            A string formatted similar to PrimitiveNode.__str__
        config_space: ConfigurationSpace
            The ConfigSpace object which defines the search space. Refer to the
            configuration/(classification||regression).py file for further details.
        strict: bool (default=True)
            Require each primitives has all required terminals present in `string`.
            Non-strict matching may be useful when constructing individuals from
            and old log with a slightly different search space.

        Returns
        -------
        PrimitiveNode
            The PrimitiveNode as defined the string.
        """
        # General form is
        # A(B(C(data[, C.param=value, ..])[, B.param=value, ..])[, A.param=value, ..])
        # below assumes that left parenthesis is never part of a parameter name or value
        primitives = string.split("(")[:-1]
        terminal_start_index = string.index(DATA_TERMINAL)
        terminals_string = string[terminal_start_index + len(DATA_TERMINAL) :]
        terminal_sets = terminals_string.split(")")[:-1]

        last_node: Union[PrimitiveNode, str] = DATA_TERMINAL
        for primitive_string, terminal_set in zip(reversed(primitives), terminal_sets):
            primitive = find_primitive(config_space, primitive_string)
            if terminal_set == "":
                terminals = []
            else:
                terminal_set = terminal_set[2:]  # 2 is because string starts with ', '
                terminals = [
                    find_terminal(config_space, terminal_string, primitive_string)
                    for terminal_string in terminal_set.split(", ")
                ]
            missing = set(primitive.input) - set(map(lambda t: t.identifier, terminals))
            if missing and strict:
                raise ValueError(f"terminals {missing} for primitive {primitive}")
            last_node = cls(primitive, last_node, terminals)

        return cast(PrimitiveNode, last_node)


def find_primitive(
    config_space: cs.ConfigurationSpace, primitive_string: str
) -> Primitive:
    """Find the Primitive that matches `primitive_string` in `config_space`."""
    if config_space is None:
        raise ValueError("config_space must not be None")
    if "estimators" not in config_space.meta:
        raise ValueError(
            "config_space must have meta information about the estimators"
            "hyperparameters"
        )

    estimators = config_space.get_hyperparameter(
        config_space.meta["estimators"]
    ).choices
    preprocessors = []
    if "preprocessors" in config_space.meta:
        preprocessors = config_space.get_hyperparameter(
            config_space.meta["preprocessors"]
        ).choices

    all_hyperparameters = estimators + preprocessors

    if primitive_string in all_hyperparameters:
        return Primitive(
            input=(),
            output=(
                "estimators" if primitive_string in estimators else "preprocessors"
            ),
            identifier=get_estimator_by_name(primitive_string),
        )

    raise IndexError(f"Could not find Primitive of type '{primitive_string}'.")


def find_terminal(
    config_space: cs.ConfigurationSpace, terminal_string: str, primitive_string: str
) -> Terminal:
    """Find the Terminal that matches `terminal_string` in `config_space`."""
    if config_space is None:
        raise ValueError("config_space must not be None")

    term_type, value = terminal_string.split("=")
    if "." in term_type:
        term_parent_type, term_type = term_type.split(".")
        term_config_space_name = f"{term_type}__{term_parent_type}"
    else:
        term_config_space_name = f"{term_type}__{primitive_string}"

    if isinstance(value, str):
        value = value.replace("'", "").replace('"', "").replace(" ", "")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = str(value)

    if term_config_space_name in config_space.get_hyperparameter_names():
        return Terminal(
            identifier=get_hyperparameter_sklearn_name(term_config_space_name),
            value=value,
            output=get_hyperparameter_sklearn_name(term_config_space_name),
            config_space_name=term_config_space_name,
        )
    if term_type in config_space.get_hyperparameter_names():
        return Terminal(
            identifier=get_hyperparameter_sklearn_name(term_type),
            value=value,
            output=get_hyperparameter_sklearn_name(term_type),
            config_space_name=term_type,
        )

    raise RuntimeError(f"Could not find Terminal of type '{terminal_string}'.")
