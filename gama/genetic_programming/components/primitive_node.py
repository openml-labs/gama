from typing import List
from .terminal import DATA_TERMINAL, Terminal
from .primitive import Primitive


class PrimitiveNode:
    """ An instantiation for a Primitive with specific Terminals. """

    def __init__(self, primitive: Primitive, data_node, terminals: List[Terminal]):
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

    def copy(self):
        """ Make a shallow copy w.r.t. Primitive/Terminal (they are immutable), but deep w.r.t. PrimitiveNodes. """
        data_node_copy = self._data_node if self._data_node == DATA_TERMINAL else self._data_node.copy()
        return PrimitiveNode(primitive=self._primitive, data_node=data_node_copy, terminals=self._terminals.copy())
