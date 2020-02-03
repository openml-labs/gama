"""
Defines the building blocks for Individuals.
Individuals represent machine learning pipelines in a back-end agnostic way.
An Individual can be converted to its back-end specific representation (e.g. a scikit-learn Pipeline
by calling its `pipeline` property as long as a function has been provided to convert the individual to it).

Individuals are built with:

 - Terminals. Definition of a specific value for a specific hyperparameter. Immutable.
 - Primitives. Definition of a specific algorithm; defined by Terminal input, output type and operation. Immutable.
 - PrimitiveNodes. An instanciated Primitive with specific Terminals. Mutable for easy operations (e.g. mutation).
 - Fitness. Stores information about the evaluation of the individual.
"""

from .primitive import Primitive
from .terminal import Terminal, DATA_TERMINAL
from .primitive_node import PrimitiveNode
from .individual import Individual
from .fitness import Fitness

__all__ = ['Individual']
