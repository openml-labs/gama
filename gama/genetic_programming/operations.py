import random
from typing import List, Optional, Union

from gama.genetic_programming.components import (
    Primitive,
    Terminal,
    PrimitiveNode,
    DATA_TERMINAL,
)


def random_terminals_for_primitive(
    primitive_set: dict, primitive: Primitive
) -> List[Terminal]:
    """Return a list with a random Terminal for each required input to Primitive."""
    return [
        random.choice([t for t in primitive_set[term_type] if isinstance(t, Terminal)])
        for term_type in primitive.input
    ]


def random_children_for_primitive(
    primitive_set: dict, primitive: Primitive, with_depth: Optional[int] = None
) -> List[Union[PrimitiveNode, Terminal]]:
    """Return a list with a random children for each required input to Primitive."""
    children = [
        random.choice(
            [
                t
                for t in primitive_set[term_type]
                if not (isinstance(t, Primitive) and with_depth == 0)
            ]
        )
        for term_type in primitive.input
    ]
    remaining_depth = with_depth - 1 if with_depth else None

    for i, child in enumerate(children):
        if isinstance(child, Primitive):
            grandchildren = random_children_for_primitive(
                primitive_set, child, with_depth=remaining_depth
            )
            children[i] = PrimitiveNode(
                child,
                data_node=DATA_TERMINAL,
                children=grandchildren,
            )
    return children


def random_primitive_node(
    output_type: str,
    primitive_set: dict,
    exclude: Optional[Primitive] = None,
    with_depth: Optional[int] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals."""
    primitive = random.choice(
        [
            p
            for p in primitive_set[output_type]
            if p != exclude and isinstance(p, Primitive)
        ]
    )
    remaining_depth = with_depth - 1 if with_depth else None
    children = random_children_for_primitive(
        primitive_set, primitive, with_depth=remaining_depth
    )
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, children=children)


def create_random_expression(
    primitive_set: dict, min_length: int = 1, max_length: int = 3
) -> PrimitiveNode:
    """Create at least min_length and at most max_length chained PrimitiveNodes."""
    individual_length = random.randint(min_length, max_length)
    learner_node = random_primitive_node(
        output_type="prediction",
        primitive_set=primitive_set,
        with_depth=individual_length,
    )
    # last_primitive_node = learner_node
    # for _ in range(individual_length - 1):
    #     primitive_node = random_primitive_node(
    #         output_type=DATA_TERMINAL, primitive_set=primitive_set
    #     )
    #     last_primitive_node._data_node = primitive_node
    #     last_primitive_node = primitive_node

    return learner_node
