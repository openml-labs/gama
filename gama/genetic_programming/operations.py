import random
from typing import List

from gama.genetic_programming.components import Primitive, Terminal, PrimitiveNode, DATA_TERMINAL, Individual


def random_terminals_for_primitive(primitive_set: dict, primitive: Primitive) -> List[Terminal]:
    """ Return a list with a Terminal matching each terminal type the Primitive requires. """
    return [random.choice(primitive_set[needed_terminal_type]) for needed_terminal_type in primitive.input]


def random_primitive_node(output_type: str, primitive_set: dict, exclude: Primitive = None) -> PrimitiveNode:
    """ Create a PrimitiveNode with a Primitive of specified output_type, with random terminals. """
    primitive = random.choice([p for p in primitive_set[output_type] if p != exclude])
    terminals = random_terminals_for_primitive(primitive_set, primitive)
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, terminals=terminals)


def create_random_expression(primitive_set: dict, min_length: int = 1, max_length: int = 3) -> PrimitiveNode:
    """ Create an individual with at least `min_length` Primitives and at most `max_length` Primitives. """
    individual_length = random.randint(min_length, max_length)
    learner_node = random_primitive_node(output_type='prediction', primitive_set=primitive_set)
    last_primitive_node = learner_node
    for _ in range(individual_length - 1):
        primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
        last_primitive_node._data_node = primitive_node
        last_primitive_node = primitive_node
    return learner_node


def create_seeded_individual(primitive_set: dict, main: Primitive, min_length: int = 1, max_length: int = 3) -> Individual:
    """ Create an individual with at least `min_length` Primitives and at most `max_length` Primitives that
        has `main` as learner node. """
    individual_length = random.randint(min_length, max_length)
    terminals = random_terminals_for_primitive(primitive_set, main)
    learner_node = PrimitiveNode(main, data_node=DATA_TERMINAL, terminals=terminals)
    last_primitive_node = learner_node

    for _ in range(individual_length - 1):
        primitive_node = random_primitive_node(output_type=DATA_TERMINAL, primitive_set=primitive_set)
        last_primitive_node._data_node = primitive_node
        last_primitive_node = primitive_node

    return Individual(learner_node)
