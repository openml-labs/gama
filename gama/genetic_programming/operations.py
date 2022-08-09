import random
from typing import List, Optional, Union
from gama.configuration.parser import compute_reachability

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
    primitive_set: dict,
    primitive: Primitive,
    with_depth: Optional[int] = None,
    skip_input_terminal: bool = False,
) -> List[Union[PrimitiveNode, Terminal]]:
    """Return a list with a random children for each required input to Primitive."""
    children = []
    for terminal_type in primitive.input:
        candidates = primitive_set[terminal_type]
        if terminal_type == primitive.data_input:
            if skip_input_terminal:
                continue
            if with_depth:
                # Since we have to adhere to a max depth,
                # we need to take into consideration required
                # preprocessing steps required to make the input data
                # fit for this primitive
                reachability = compute_reachability(primitive_set)
                if reachability[primitive.data_input] == with_depth:
                    candidates = [
                        c
                        for c in candidates
                        if isinstance(c, Primitive)
                        and reachability[c.data_input] == with_depth - 1
                    ]
            elif with_depth == 0:
                candidates = [c for c in candidates if isinstance(c, Terminal)]

        children.append(random.choice(candidates))
    # children = [
    #     random.choice(
    #         [
    #             t
    #             for t in primitive_set[term_type]
    #             if not (isinstance(t, Primitive) and with_depth == 0)
    #         ]
    #     )
    #     for term_type in primitive.input
    #     if not skip_input_terminal or term_type != primitive.data_input
    # ]
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
    skip_input_terminal: bool = False,
    with_depth: Optional[int] = None,
    data_input_type: Optional[str] = None,
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals."""
    candidates = [
        p
        for p in primitive_set[output_type]
        if p != exclude and isinstance(p, Primitive)
    ]
    if data_input_type:
        candidates = [c for c in candidates if c.data_input == data_input_type]
    elif with_depth:
        reachability = compute_reachability(primitive_set)
        # if the maximum depth is exactly the minimum number of steps
        # we need to the input data then we need to make sure we take
        # a step that brings us closer to the input data
        if with_depth == reachability[output_type]:
            candidates = [
                c for c in candidates if reachability[c.data_input] == with_depth - 1
            ]

    primitive = random.choice(candidates)
    remaining_depth = with_depth - 1 if with_depth else None
    children = random_children_for_primitive(
        primitive_set,
        primitive,
        with_depth=remaining_depth,
        skip_input_terminal=skip_input_terminal,
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

    return learner_node
