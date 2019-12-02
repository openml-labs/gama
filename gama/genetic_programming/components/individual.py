import uuid
from typing import List, Callable, Optional
from .primitive_node import PrimitiveNode
from .terminal import DATA_TERMINAL, Terminal


class Individual:
    """ A collection of PrimitiveNodes which together specify a machine learning pipeline.

    Parameters
    ----------
    main_node: PrimitiveNode
        The first node of the individual (the estimator node).
    to_pipeline: Callable, optional (default=None)
         A function which can convert this individual into a machine learning pipeline.
         If not provided, the `pipeline` property will be unavailable.
    """

    def __init__(self, main_node: PrimitiveNode, to_pipeline: Optional[Callable] = None):
        self.fitness = None
        self.main_node = main_node
        self._id = uuid.uuid4()
        self._to_pipeline = to_pipeline

    def __eq__(self, other):
        return isinstance(other, Individual) and other._id == self._id

    def __str__(self):
        return """Individual {}\nPipeline: {}\nFitness: {}""".format(self._id, self.pipeline_str(), self.fitness)

    @property
    def pipeline(self):
        """ Calls the `to_pipeline` method on itself. Typically a scikit-learn Pipeline. """
        if self._to_pipeline is None:
            raise AttributeError("`pipeline` not available because `to_pipeline` was not set on __init__.")
        return self._to_pipeline(self)

    def short_name(self, step_separator: str = '>'):
        """ str: e.g. "Binarizer>BernoulliNB" """
        return step_separator.join([str(primitive._primitive) for primitive in reversed(self.primitives)])

    def pipeline_str(self):
        """ str: e.g. "BernoulliNB(Binarizer(data, Binarizer.threshold=0.6), BernoulliNB.alpha=1.0)" """
        return str(self.main_node)

    @property
    def primitives(self) -> List[PrimitiveNode]:
        """ List[PrimitiveNode]: lists all primitive nodes, starting with the Individual's main node. """
        primitives = [self.main_node]
        current_node = self.main_node._data_node
        while current_node != DATA_TERMINAL:
            primitives.append(current_node)
            current_node = current_node._data_node
        return primitives

    @property
    def terminals(self) -> List[Terminal]:
        """ List[Terminal]: lists all terminals connected to the Individual's primitive nodes. """
        return [terminal for primitive in self.primitives for terminal in primitive._terminals]

    def replace_terminal(self, position: int, new_terminal: Terminal):
        """ Replace the terminal at `position` by `new_terminal` in-place.

        Parameters
        ----------
        position: int
            Position (in self.terminals) of the terminal to be replaced.
        new_terminal: Terminal
            The new terminal to replace the old one with. Must share output type.
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

        Parameters
        ----------
        position: int
            Position (in self.primitives) of the PrimitiveNode to be replaced.
        new_primitive: PrimitiveNode
            The new PrimitiveNode to replace the old one with. Must share output type.
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
        """ Make a deep copy of the individual, but with fitness set to None and assigned a new id. """
        return Individual(main_node=self.main_node.copy(), to_pipeline=self._to_pipeline)

    @classmethod
    def from_string(cls, string: str, primitive_set: dict, to_pipeline: Optional[Callable] = None):
        """ Construct an Individual from its `pipeline_str` representation.

        Parameters
        ----------
        string: str
            String formatted as `Individual.pipeline_str`.
        primitive_set: dict
            The dictionary defining all Terminals and Primitives.
        to_pipeline: Callable, optional (default=None)
            The function to convert the Individual into a pipeline representation.
            If `None`, the individuals `pipeline` property will not be available.

        Returns
        -------
        Individual
            An individual as defined by `str`.
        """
        expression = PrimitiveNode.from_string(string, primitive_set)
        return cls(expression, to_pipeline=to_pipeline)
