import uuid
from typing import List, Callable, Optional, Dict, Any

from sklearn.pipeline import Pipeline

from .fitness import Fitness
from .primitive_node import PrimitiveNode
from .terminal import Terminal


class Individual:
    """Collection of PrimitiveNodes which together specify a machine learning pipeline.

    Parameters
    ----------
    main_node: PrimitiveNode
        The first node of the individual (the estimator node).
    to_pipeline: Callable, optional (default=None)
         A function which can convert this individual into a machine learning pipeline.
         If not provided, the `pipeline` property will be unavailable.
    """

    def __init__(
        self, main_node: PrimitiveNode, to_pipeline: Optional[Callable] = None
    ):
        self.fitness: Optional[Fitness] = None
        self.main_node = main_node
        self.meta: Dict[str, Any] = dict()
        self._id = uuid.uuid4()
        self._to_pipeline = to_pipeline

    def __eq__(self, other) -> bool:
        return isinstance(other, Individual) and other._id == self._id

    def __hash__(self) -> int:
        return hash(self._id)

    def __str__(self) -> str:
        return (
            f"Individual {self._id}\n"
            f"Pipeline: {self.pipeline_str()}\nFitness: {self.fitness}"
        )

    @property
    def pipeline(self) -> Pipeline:
        """Calls the `to_pipeline` method on itself."""
        if self._to_pipeline is None:
            raise AttributeError(
                "pipeline not available because `to_pipeline` was not set on __init__."
            )
        return self._to_pipeline(self)

    def short_name(self, step_separator: str = ">") -> str:
        """str: e.g. "Binarizer>BernoulliNB" """
        return step_separator.join(
            [str(primitive._primitive) for primitive in reversed(self.primitives)]
        )

    def pipeline_str(self) -> str:
        """str: e.g., "BernoulliNB(Binarizer(data, Binarizer.threshold=0.6), BernoulliNB.alpha=1.0)" """  # noqa: E501
        return str(self.main_node)

    @property
    def primitives(self) -> List[PrimitiveNode]:
        """Lists all primitive nodes, starting with the Individual's main node."""

        def is_data_primitive(child) -> bool:
            return isinstance(child, PrimitiveNode) and child._primitive.data_input != "dont_remove"

        primitives = [self.main_node]
        current_children = self.main_node._children
        while any(is_data_primitive(child) for child in current_children):
            # Only data input can be a primitive node, so there is never more than one.
            child_node = next(
                child for child in current_children
                if is_data_primitive(child)
            )
            primitives.append(child_node)
            current_children = child_node._children
        return primitives

    @property
    def terminals(self) -> List[Terminal]:
        """Lists all terminals connected to the Individual's primitive nodes."""
        return [terminal for prim in self.primitives for terminal in prim.terminals]

    def replace_terminal(self, position: int, new_terminal: Terminal) -> None:
        """Replace the terminal at `position` by `new_terminal` in-place.

        Parameters
        ----------
        position: int
            Position (in self.terminals) of the terminal to be replaced.
        new_terminal: Terminal
            The new terminal to replace the old one with. Must share output type.
        """
        scan_position = 0
        for primitive in self.primitives:
            if scan_position + len(primitive.terminals) > position:
                terminal_to_be_replaced = primitive.terminals[position - scan_position]
                if terminal_to_be_replaced.identifier == new_terminal.identifier:
                    i = primitive._children.index(terminal_to_be_replaced)
                    primitive._children[i] = new_terminal
                    return
                else:
                    raise ValueError(
                        f"New terminal does not share output type with the old. "
                        f"Old: {terminal_to_be_replaced.identifier}"
                        f"New: {new_terminal.identifier}."
                    )
            else:
                scan_position += len(primitive.terminals)
        if scan_position < position:
            raise ValueError(
                f"Position {position} is out of range with {scan_position} terminals."
            )

    def replace_primitive(self, position: int, new_primitive: PrimitiveNode):
        """Replace the PrimitiveNode at `position` by `new_primitive`.

        Parameters
        ----------
        position: int
            Position (in self.primitives) of the PrimitiveNode to be replaced.
        new_primitive: PrimitiveNode
            The new PrimitiveNode to replace the old one with. Must share output type.
        """
        to_be_replaced = self.primitives[position]

        if to_be_replaced._primitive.output != new_primitive._primitive.output:
            raise ValueError("New primitive output type differs from old.")
        if to_be_replaced.input_node:
            new_primitive.replace_or_add_input_node(to_be_replaced.input_node)
        if position == 0:
            self.main_node = new_primitive
        else:
            self.primitives[position - 1].replace_or_add_input_node(new_primitive)

    def copy_as_new(self) -> "Individual":
        """Make deep copy of self, but with fitness None and assigned a new id."""
        return Individual(self.main_node.copy(), to_pipeline=self._to_pipeline)

    @classmethod
    def from_string(
        cls, string: str, primitive_set: dict, to_pipeline: Optional[Callable] = None
    ) -> "Individual":
        """Construct an Individual from its `pipeline_str` representation.

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
