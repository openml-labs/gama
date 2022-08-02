import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
from gama.genetic_programming.components.primitive_node import PrimitiveNode

from sklearn.pipeline import Pipeline

from gama.utilities.evaluation_library import Evaluation

from .components import Individual

log = logging.getLogger(__name__)


class OperatorSet:
    """Provides a thin layer for ea operators for logging, callbacks and safety."""

    def __init__(
        self,
        mutate: Callable[[Individual], None],
        mate: Callable[[Individual, Individual], Tuple[Individual, Individual]],
        create_from_population: Callable[[Any], List[Individual]],
        create_new: Callable[[], PrimitiveNode],
        compile_: Callable[[Individual], Pipeline],
        eliminate: Callable[[List[Individual], int], List[Individual]],
        evaluate_callback: Callable[[Evaluation], None],
        max_retry: int = 50,
        completed_evaluations: Optional[Dict[str, Evaluation]] = None,
    ):
        self._mutate = mutate
        self._mate = mate
        self._create_from_population = create_from_population
        self._create_new = create_new
        self._compile = compile_
        self._safe_compile: Optional[Callable[[Individual], Pipeline]] = None
        self._eliminate = eliminate
        self._max_retry = max_retry
        self._evaluate = None
        self._evaluate_callback = evaluate_callback
        self.evaluate: Optional[Callable[..., Evaluation]] = None

        self._completed_evaluations = completed_evaluations

    def wait_next(self, async_evaluator):
        """Wrapper for wait_next() to forward evaluation and log exceptions."""
        future = async_evaluator.wait_next()
        if future.result is not None:
            evaluation = future.result
            if self._evaluate_callback is not None:
                self._evaluate_callback(evaluation)

        elif future.exception is not None:
            log.warning(f"Error raised during evaluation: {str(future.exception)}.")
        return future

    def try_until_new(self, operator, *args, **kwargs):
        """Keep executing `operator` until a new individual is created."""
        for _ in range(self._max_retry):
            individual = operator(*args, **kwargs)
            if str(individual.main_node) not in self._completed_evaluations:
                return individual
        else:
            log.debug(f"50 iterations of {operator.__name__} did not yield new ind.")
            # For progress on solving this, see #11
            return individual

    def mate(self, ind1: Individual, ind2: Individual, *args, **kwargs) -> Individual:
        def mate_with_log():
            new_individual1, new_individual2 = ind1.copy_as_new(), ind2.copy_as_new()
            self._mate(new_individual1, new_individual2, *args, **kwargs)
            new_individual1.meta = dict(parents=[ind1._id, ind2._id], origin="cx")
            return new_individual1

        return self.try_until_new(mate_with_log)

    def mutate(self, ind: Individual, *args, **kwargs) -> Individual:
        def mutate_with_log():
            new_individual = ind.copy_as_new()
            mutator = self._mutate(new_individual, *args, **kwargs)
            new_individual.meta = dict(parents=[ind._id], origin=mutator.__name__)
            return new_individual

        return self.try_until_new(mutate_with_log)

    def individual(self, *args, **kwargs) -> Individual:
        expression = self._create_new(*args, **kwargs)
        if self._safe_compile is not None:
            compile_ = self._safe_compile
        else:
            compile_ = self._compile

        ind = Individual(expression, to_pipeline=compile_)
        ind.meta["origin"] = "new"
        return ind

    def create(self, *args, **kwargs) -> List[Individual]:
        return self._create_from_population(self, *args, **kwargs)

    def eliminate(self, *args, **kwargs):
        return self._eliminate(*args, **kwargs)
