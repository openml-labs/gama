import logging

from .components import Individual

log = logging.getLogger(__name__)


class OperatorSet:
    """ Provides a thin layer for ea operators for logging, callbacks and safety. """

    def __init__(
        self,
        mutate,
        mate,
        create_from_population,
        create_new,
        compile_,
        eliminate,
        evaluate_callback,
        max_retry=50,
        completed_evaluations=None,
    ):
        """

        :param mutate:
        :param mate:
        :param create:
        :param create_new:
        """

        self._mutate = mutate
        self._mate = mate
        self._create_from_population = create_from_population
        self._create_new = create_new
        self._compile = compile_
        self._safe_compile = None
        self._eliminate = eliminate
        self._max_retry = max_retry
        self._evaluate = None
        self._evaluate_callback = evaluate_callback
        self.evaluate = None

        self._completed_evaluations = completed_evaluations

    def wait_next(self, async_evaluator):
        future = async_evaluator.wait_next()
        if future.result is not None:
            evaluation = future.result
            if self._evaluate_callback is not None:
                self._evaluate_callback(evaluation)

        elif future.exception is not None:
            log.warning(f"Error raised during evaluation: {str(future.exception)}.")
        return future

    def try_until_new(self, operator, *args, **kwargs):
        for _ in range(self._max_retry):
            individual = operator(*args, **kwargs)
            if str(individual.main_node) not in self._completed_evaluations:
                return individual
        else:
            log.debug(f"50 iterations of {operator.__name__} did not yield new ind.")
            # For progress on solving this, see #11
            return individual

    def mate(self, ind1: Individual, ind2: Individual, *args, **kwargs):
        def mate_with_log():
            new_individual1, new_individual2 = ind1.copy_as_new(), ind2.copy_as_new()
            self._mate(new_individual1, new_individual2, *args, **kwargs)
            new_individual1.meta = dict(parents=[ind1._id, ind2._id], origin="cx")
            return new_individual1

        individual = self.try_until_new(mate_with_log)
        return individual

    def mutate(self, ind: Individual, *args, **kwargs):
        def mutate_with_log():
            new_individual = ind.copy_as_new()
            mutator = self._mutate(new_individual, *args, **kwargs)
            new_individual.meta = dict(parents=[ind._id], origin=mutator.__name__)
            return new_individual

        ind = self.try_until_new(mutate_with_log)
        return ind

    def individual(self, *args, **kwargs):
        expression = self._create_new(*args, **kwargs)
        if self._safe_compile is not None:
            compile_ = self._safe_compile
        else:
            compile_ = self._compile
        ind = Individual(expression, to_pipeline=compile_)
        ind.meta["origin"] = "new"
        return ind

    def create(self, *args, **kwargs):
        return self._create_from_population(self, *args, **kwargs)

    def eliminate(self, *args, **kwargs):
        return self._eliminate(*args, **kwargs)
