"""
Defines the building blocks for individuals, in short these are the types:
- Terminal. Definition of a specific value for a specific hyperparameter. Immutable.
- Primitive. Definition of a specific algorithm; defined by Terminal input, output type and operation. Immutable.
- PrimitiveNode. An instanciated Primitive with specific Terminals. Mutable for easy operations (e.g. mutation).
- Individual. A sequence of PrimitiveNodes that together form a pipeline.
    - Fitness. Stores information about the evaluation of the individual.

"""

from .primitive import Primitive
from .terminal import Terminal, DATA_TERMINAL
from .primitive_node import PrimitiveNode
from .individual import Individual
from .fitness import Fitness