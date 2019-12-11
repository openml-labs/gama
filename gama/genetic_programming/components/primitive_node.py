from typing import List
from .terminal import DATA_TERMINAL, Terminal
from .primitive import Primitive


class PrimitiveNode:
    """ An instantiation for a Primitive with specific Terminals.

    Parameters
    ----------
    primitive: Primitive
        The Primitive type of this PrimitiveNode.
    data_node: PrimitiveNode
        The PrimitiveNode that specifies all preprocessing before this PrimitiveNode.
    terminals: List[Terminal]
        A list of terminals matching the `primitive`.
    """

    def __init__(self, primitive: Primitive, data_node: 'PrimitiveNode', terminals: List[Terminal]):
        self._primitive = primitive
        self._data_node = data_node
        self._terminals = sorted(terminals, key=lambda t: str(t))

    def __str__(self):
        """ Recursively stringify all primitive nodes (primitive and hyperparameters).

        Examples: - "GaussianNB(data)"
                  - "BernoulliNB(data, alpha=1.0)"
                  - "BernoulliNB(FastICA(data, tol=0.5), alpha=1.0)"
        """
        if self._terminals:
            terminal_str = ", ".join([repr(terminal) for terminal in self._terminals])
            return "{}({}, {})".format(self._primitive, str(self._data_node), terminal_str)
        else:
            return "{}({})".format(self._primitive, str(self._data_node))

    @property
    def str_nonrecursive(self) -> str:
        """ Stringify all primitive node without data node (primitive and hyperparameters).

        Examples: - "GaussianNB()"
                  - "BernoulliNB(alpha=1.0)"
        """
        terminal_str = ", ".join([str(terminal) for terminal in self._terminals])
        return "{}({})".format(self._primitive, terminal_str)

    def copy(self):
        """ Make a shallow copy w.r.t. Primitive/Terminal (they are immutable), but deep w.r.t. PrimitiveNodes. """
        data_node_copy = self._data_node if self._data_node == DATA_TERMINAL else self._data_node.copy()
        return PrimitiveNode(primitive=self._primitive, data_node=data_node_copy, terminals=self._terminals.copy())

    @classmethod
    def from_string(cls, string: str, primitive_set: dict):
        """ Create a PrimitiveNode from string formatted like PrimitiveNode.__str__

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
        # General form is A(B(C(data[, C.param=value, ...])[, B.param=value, ...])[, A.param=value, ...])
        # below assumes that left parenthesis is never part of a parameter name or value.
        primitives = string.split('(')[:-1]
        terminal_start_index = string.index(DATA_TERMINAL)
        terminals_string = string[terminal_start_index + len(DATA_TERMINAL):]
        terminal_sets = terminals_string.split(')')[:-1]

        last_node = DATA_TERMINAL
        for primitive_string, terminal_set in zip(reversed(primitives), terminal_sets):
            primitive = find_primitive(primitive_set, primitive_string)
            if terminal_set == '':
                terminals = []
            else:
                terminal_set = terminal_set[2:]  # 2 is because string starts with ', '
                terminals = [find_terminal(primitive_set, terminal_string)
                             for terminal_string in terminal_set.split(', ')]
            if not all([required_terminal in map(lambda t: t.identifier, terminals)
                        for required_terminal in primitive.input]):
                missing = [required_terminal for required_terminal in primitive.input
                           if required_terminal not in map(lambda t: t.identifier, terminals)]
                raise ValueError("Individual does not define all required terminals for primitive {}. Missing: {}."
                                 .format(primitive, missing))
            last_node = cls(primitive, last_node, terminals)

        return last_node


def find_primitive(primitive_set: dict, primitive_string: str) -> Primitive:
    """ Find the Primitive that matches `primitive_string` in `primitive_set`. """
    all_primitives = primitive_set[DATA_TERMINAL] + primitive_set['prediction']
    try:
        return [p for p in all_primitives if repr(p) == primitive_string][0]
    except IndexError:
        raise IndexError(f"Could not find Primitive of type '{primitive_string}'.")



def find_terminal(primitive_set: dict, terminal_string: str) -> Terminal:
    """ Find the Terminal that matches `terminal_string` in `primitive_set`. """
    terminal_return_type, terminal_value = terminal_string.split('=')
    try:
        return [t for t in primitive_set[terminal_return_type] if repr(t) == terminal_string][0]
    except IndexError:
        raise IndexError(f"Could not find Terminal of type '{terminal_string}'.")
