import uuid
from typing import List
from .primitive_node import PrimitiveNode
from .terminal import DATA_TERMINAL, Terminal
from .primitive import Primitive


class Individual:
    """ A collection of PrimitiveNodes which together specify a machine learning pipeline. """

    def __init__(self, main_node: PrimitiveNode):
        self.fitness = None
        self.main_node = main_node
        self._id = uuid.uuid4()

    def pipeline_str(self):
        """ e.g. "BernoulliNB(Binarizer(data, Binarizer.threshold=0.6), BernoulliNB.alpha=1.0)" """
        return str(self.main_node)

    def __eq__(self, other):
        return isinstance(other, Individual) and other._id == self._id

    def __str__(self):
        return """Individual {}\nPipeline: {}\nFitness: {}""".format(self._id, self.pipeline_str(), self.fitness)

    @property
    def primitives(self) -> List[PrimitiveNode]:
        primitives = [self.main_node]
        current_node = self.main_node._data_node
        while current_node != DATA_TERMINAL:
            primitives.append(current_node)
            current_node = current_node._data_node
        return primitives

    @property
    def terminals(self) -> List[Terminal]:
        return [terminal for primitive in self.primitives for terminal in primitive._terminals]

    def replace_terminal(self, position: int, new_terminal: Terminal):
        """ Replace the terminal at `position` by `new_terminal`.

        The old terminal is the one found at `position` in self.terminals.
        The `new_terminal` and old terminal must share output type.
        """
        scan_position = 0
        for primitive in self.primitives:
            if scan_position + len(primitive._terminals) > position:
                terminal_to_be_replaced = primitive._terminals[position - scan_position]
                if terminal_to_be_replaced.identifier == new_terminal.identifier:
                    primitive._terminals[position - scan_position] = new_terminal
                    return
                else:
                    raise ValueError("New terminal does not share output type with the one at position {}."
                                     "Old: {}. New: {}.".format(position,
                                                                terminal_to_be_replaced.identifier,
                                                                new_terminal.identifier))
            else:
                scan_position += len(primitive._terminals)
        if scan_position < position:
            raise ValueError("Position {} is out of range with {} terminals.".format(position, scan_position))

    def replace_primitive(self, position: int, new_primitive: PrimitiveNode):
        """ Replace the PrimitiveNode at `position` by `new_primitive`.

        The old PrimitiveNode is the one found at `position` in self.primitives.
        The `new_primitive` and old PrimitiveNode must share output type.
        """
        last_primitive = None
        for i, primitive_node in enumerate(self.primitives):
            if i == position:
                if primitive_node._primitive.output != new_primitive._primitive.output:
                    raise ValueError("New primitive does not produce same output as the primitive to be replaced.")
                if isinstance(primitive_node._data_node, str):
                    new_primitive._data_node = primitive_node._data_node
                else:
                    new_primitive._data_node = primitive_node._data_node.copy()
                break
            else:
                last_primitive = primitive_node

        if position == 0:
            self.main_node = new_primitive
        else:
            last_primitive._data_node = new_primitive

    def copy_as_new(self):
        """ Make a deep copy of the individual, but with fitness set to None and assign a new id. """
        return Individual(main_node=self.main_node.copy())

    def can_mate_with(self, other) -> bool:
        """ True if `self` and `other` share at least one primitive or both have at least two primitives, else false."""
        other_primitives = list(map(lambda primitive_node: primitive_node._primitive, other.primitives))
        # Shared primitives mean they can exchange terminals
        shared_primitives = [p for p in self.primitives if p._primitive in other_primitives]
        # Both at least two primitives means they can swap primitives
        both_at_least_length_2 = len(other_primitives) >= 2 and len(self.primitives) >= 2
        return both_at_least_length_2 or shared_primitives

    @classmethod
    def from_string(cls, string: str, primitive_set: dict):
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
            last_node = PrimitiveNode(primitive, last_node, terminals)

        return cls(last_node)


def find_primitive(primitive_set: dict, primitive_string: str) -> Primitive:
    """ Find the Primitive that matches `primitive_string` in `primitive_set`. """
    all_primitives = primitive_set[DATA_TERMINAL] + primitive_set['prediction']
    results = [p for p in all_primitives if repr(p) == primitive_string]
    if results == []:
        raise ValueError("Primitive {} not found in primitive set".format(primitive_string))
    return results[0]


def find_terminal(primitive_set: dict, terminal_string: str) -> Terminal:
    """ Find the Terminal that matches `terminal_string` in `primitive_set`. """
    terminal_return_type, terminal_value = terminal_string.split('=')
    return [t for t in primitive_set[terminal_return_type] if repr(t) == terminal_string][0]
