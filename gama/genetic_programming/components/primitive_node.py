import itertools
from typing import List, Optional, Sequence, Union, cast
from .terminal import Terminal
from .primitive import Primitive


class PrimitiveNode:
    """An instantiation for a Primitive with specific Terminals.

    Parameters
    ----------
    primitive: Primitive
        The Primitive type of this PrimitiveNode.
    data_node: PrimitiveNode
        The PrimitiveNode that specifies all preprocessing before this PrimitiveNode.
    children: List[Union["PrimitiveNode", Terminal]]
        A non-empty list of terminals and primitivenodes matching the `primitive` input.
    """

    def __init__(
        self,
        primitive: Primitive,
        data_node: Union["PrimitiveNode", str],
        children: Sequence[Union["PrimitiveNode", Terminal]],
    ):
        self._primitive = primitive
        self._data_node = data_node
        self._children = sorted(children, key=lambda t: str(t))

    def __str__(self) -> str:
        """Recursively stringify all primitive nodes (primitive and hyperparameters).

        Examples: - "GaussianNB(data)"
                  - "BernoulliNB(data, alpha=1.0)"
                  - "BernoulliNB(FastICA(data, tol=0.5), alpha=1.0)"
        """
        input_str = f"{self.input_node!r}" if self.input_node else ""
        terminal_str = ", ".join(
            [
                repr(terminal) if isinstance(terminal, Terminal) else str(terminal)
                for terminal in self.terminals + self.primitives
                if terminal != self.input_node
            ]
        )
        join = ", " if input_str and terminal_str else ""
        return f"{self._primitive}({input_str}{join}{terminal_str})"

    def __repr__(self) -> str:
        return str(self)

    @property
    def input_node(self) -> Optional[Union[Terminal, "PrimitiveNode"]]:
        """Returns the child that provides the input data."""
        return next(
            (
                c
                for c in self._children
                if (isinstance(c, Terminal) and c.output == self._primitive.data_input)
                or (
                    isinstance(c, PrimitiveNode)
                    and c._primitive.output == self._primitive.data_input
                )
            ),
            None,
        )

    def replace_or_add_input_node(
        self, new_node: Union[None, Terminal, "PrimitiveNode"]
    ) -> None:
        """Replace the input node with the provided node."""
        current_node = self.input_node
        if current_node:
            self._children.remove(current_node)
        if new_node:
            self._children.append(new_node)

    @property
    def primitives(self) -> List["PrimitiveNode"]:
        """Returns all children of this node that are terminals."""
        return [c for c in self._children if isinstance(c, PrimitiveNode)]

    @property
    def terminals(self) -> List[Terminal]:
        """Returns all children of this node that are terminals."""
        return [c for c in self._children if isinstance(c, Terminal)]

    @property
    def str_nonrecursive(self) -> str:
        """Stringify primitive node with its hyperparameter configuration

        Examples: - "GaussianNB()"
                  - "BernoulliNB(alpha=1.0)"
        """
        terminal_str = ", ".join(
            [
                str(terminal)
                for terminal in self._children
                if terminal != self.input_node
            ]
        )
        return f"{self._primitive}({terminal_str})"

    def copy(self) -> "PrimitiveNode":
        """Copies the object. Shallow for terminals, deep for data_node."""
        children: List[Union["PrimitiveNode", Terminal]] = [
            child.copy() if isinstance(child, PrimitiveNode) else child
            for child in self._children
        ]
        return PrimitiveNode(
            primitive=self._primitive,
            data_node=self._data_node,
            children=children,
        )

    @classmethod
    def from_string(cls, string: str, primitive_set: dict) -> "PrimitiveNode":
        """Create a PrimitiveNode from string formatted like PrimitiveNode.__str__

        Parameters
        ----------
        string: str
            A string formatted similar to PrimitiveNode.__str__
        primitive_set: dict
            The dictionary defining all Terminals and Primitives.

        Returns
        -------
        PrimitiveNode
            The PrimitiveNode as defined the string.
        """
        # General form is
        # A(B(C(data[, C.param=value, ..])[, B.param=value, ..])[, A.param=value, ..])
        # below assumes that left parenthesis is never part of a parameter name or value
        primitives = string.split("(")[:-1]
        terminal_start_index = string.rindex("(") + 1
        terminals_string = string[terminal_start_index:]
        terminal_sets = terminals_string.split(")")[:-1]

        last_node: Optional[PrimitiveNode] = None
        for primitive_string, terminal_set in zip(reversed(primitives), terminal_sets):
            primitive = find_primitive(primitive_set, primitive_string)
            if terminal_set == "":
                terminals = []
            else:
                if terminal_set.startswith(", "):
                    terminal_set = terminal_set[2:]
                terminals = [
                    find_terminal(primitive_set, terminal_string)
                    for terminal_string in terminal_set.split(", ")
                ]
            missing = (
                set(primitive.input)
                - set(map(lambda t: t.identifier, terminals))
                - {last_node._primitive.output}
                if last_node
                else {}
            )
            if missing:
                raise ValueError(f"terminals {missing} for primitive {primitive}")
            children: List[Union[Terminal, "PrimitiveNode"]] = terminals  # type: ignore
            if last_node:
                children.append(last_node)
            last_node = cls(primitive, last_node, children)  # type: ignore

        return cast(PrimitiveNode, last_node)


def find_primitive(primitive_set: dict, primitive_string: str) -> Primitive:
    """Find the Primitive that matches `primitive_string` in `primitive_set`."""
    all_primitives = [
        i for i in itertools.chain(*primitive_set.values()) if isinstance(i, Primitive)
    ]
    for primitive in all_primitives:
        if repr(primitive) == primitive_string:
            return primitive
    raise IndexError(f"Could not find Primitive of type '{primitive_string}'.")


def find_terminal(primitive_set: dict, terminal_string: str) -> Terminal:
    """Find the Terminal that matches `terminal_string` in `primitive_set`."""
    # BANDAGE: this is for backwards compatibility - remove and revise the test suite?
    if terminal_string == "data":
        terminal_string = "numeric_data='data'"
    term_type, _ = terminal_string.split("=")
    for terminal in primitive_set[term_type]:
        if repr(terminal) == terminal_string:
            return terminal
    raise RuntimeError(f"Could not find Terminal of type '{terminal_string}'.")
