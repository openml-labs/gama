from collections import defaultdict
import random
from typing import List, Callable, NamedTuple
import uuid

import sklearn

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

# Defines an operator which takes input and produces output, e.g. a preprocessing or classification algorithm.
Primitive = NamedTuple("Primitive",
                       [("input", List[str]),
                        ("output", str),
                        ("identifier", Callable)])


def primitive__str__(primitive) -> str:
    """ e.g. "FastICA" """
    return primitive.identifier.__name__


def primitive__repr__(primitive) -> str:
    """ e.g. "FastICA" """
    return primitive.identifier.__name__


Primitive.__str__ = primitive__str__
Primitive.__repr__ = primitive__repr__


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


class Fitness:
    def __init__(self):
        self.values = None
        self.start_time = None
        self.time = None


class Individual:
    """ A collection of PrimitiveNodes which together specify a machine learning pipeline. """

    def __init__(self, main_node: PrimitiveNode):
        self.fitness = Fitness()
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


def find_primitive(primitive_set: dict, primitive_string: str):
    all_primitives = primitive_set[DATA_TERMINAL] + primitive_set['prediction']
    return [p for p in all_primitives if repr(p) == primitive_string][0]


def find_terminal(primitive_set: dict, terminal_string: str):
    terminal_return_type, terminal_value = terminal_string.split('=')
    return [t for t in primitive_set[terminal_return_type] if repr(t) == terminal_string][0]


def pset_from_config2(configuration):
    """ Create a pset for the given configuration dictionary.

    Given a configuration dictionary specifying operators (e.g. sklearn
    estimators), their hyperparameters and values for each hyperparameter,
    create a gp.PrimitiveSetTyped that contains:
        - For each operator a primitive
        - For each possible hyperparameter-value combination a unique terminal

    Side effect: Imports the classes of each primitive.

    Returns the given Pset.
    """

    pset = defaultdict(list)
    parameter_checks = {}

    shared_hyperparameter_types = {}
    # We have to make sure the str-keys are evaluated first: they describe shared hyperparameters
    # We can not rely on order-preserving dictionaries as this is not in the Python 3.5 specification.
    sorted_keys = reversed(sorted(configuration.keys(), key=lambda x: str(type(x))))
    for key in sorted_keys:
        values = configuration[key]
        if isinstance(key, str):
            # Specification of shared hyperparameters
            for value in values:
                pset[key].append(Terminal(value=value, output=key, identifier=key))
        elif isinstance(key, object):
            # Specification of operator (learner, preprocessor)
            hyperparameter_types = []
            for name, param_values in sorted(values.items()):
                # We construct a new type for each hyperparameter, so we can specify
                # it as terminal type, making sure it matches with expected
                # input of the operators. Moreover it automatically makes sure that
                # crossover only happens between same hyperparameters.
                if isinstance(param_values, list) and not param_values:
                    # An empty list indicates a shared hyperparameter
                    hyperparameter_types.append(name)
                elif name == "param_check":
                    # This allows users to define illegal hyperparameter combinations, but is not a terminal.
                    parameter_checks[key.__name__] = param_values[0]
                else:
                    hyperparameter_types.append(key.__name__ + '.' + name)
                    for value in param_values:
                        pset[key.__name__ + '.' + name].append(
                            Terminal(value=value, output=name, identifier=key.__name__ + '.' + name))

            # After registering the hyperparameter types, we can register the operator itself.
            transformer_tags = ["DATA_PREPROCESSING", "FEATURE_SELECTION", "DATA_TRANSFORMATION"]
            if (issubclass(key, sklearn.base.TransformerMixin) or
                    (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] in transformer_tags)):
                pset[DATA_TERMINAL].append(Primitive(input=hyperparameter_types, output=DATA_TERMINAL, identifier=key))
            elif (issubclass(key, sklearn.base.ClassifierMixin) or
                  (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] == "CLASSIFICATION")):
                pset["prediction"].append(Primitive(input=hyperparameter_types, output="prediction", identifier=key))
            elif (issubclass(key, sklearn.base.RegressorMixin) or
                  (hasattr(key, 'metadata') and key.metadata.query()["primitive_family"] == "REGRESSION")):
                pset["prediction"].append(Primitive(input=hyperparameter_types, output="prediction", identifier=key))
            else:
                raise TypeError("Expected {} to be either subclass of "
                                "TransformerMixin, RegressorMixin or ClassifierMixin.".format(key))
        else:
            raise TypeError('Encountered unknown type as key in dictionary.'
                            'Keys in the configuration should be str or class.')

    return pset, parameter_checks


def random_terminals_for_primitive(primitive_set: dict, primitive: Primitive):
    return [random.choice(primitive_set[needed_terminal_type]) for needed_terminal_type in primitive.input]


def random_primitive_node(output_type: str, primitive_set: dict, exclude: Primitive = None):
    """ Create a PrimitiveNode with a Primitive of specified output_type, with random terminals. """
    primitive = random.choice([p for p in primitive_set[output_type] if p != exclude])
    terminals = random_terminals_for_primitive(primitive_set, primitive)
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, terminals=terminals)


def create_random_individual(primitive_set: dict, min_length: int = 1, max_length: int = 3) -> Individual:
    individual_length = random.randint(min_length, max_length)
    learner_node = random_primitive_node(output_type='prediction', primitive_set=primitive_set)
    last_primitive_node = learner_node
    for _ in range(individual_length - 1):
        primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
        last_primitive_node._data_node = primitive_node
        last_primitive_node = primitive_node

    return Individual(learner_node)
