from collections import Sequence
import logging

from gama.logging.machine_logging import TOKENS, log_event
from gama.utilities.generic.async_executor import wait_first_complete
from .components import Individual

log = logging.getLogger(__name__)


class OperatorSet:
    """ Provides a thin layer for ea operators for logging, callbacks and safety. """

    def __init__(self, mutate, mate, create_from_population, create_new, compile_, eliminate, evaluate_callback, max_retry=50):
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

        self._seen_individuals = {}

    def wait_first_complete(self, *args, **kwargs):
        done, not_done = wait_first_complete(*args, **kwargs)
        for result in [future.result() for future in done]:
            individual = result if not isinstance(result, Sequence) else result[0]
            if self._evaluate_callback is not None:
                self._evaluate_callback(individual)
        return done, not_done

    def try_until_new(self, operator, *args, **kwargs):
        for _ in range(self._max_retry):
            individual, log_args = operator(*args, **kwargs)
            if str(individual.main_node) not in self._seen_individuals:
                #self._seen_individuals[str(individual.main_node)] = individual
                return individual, log_args
        else:
            log.debug("Could not create a new individual from 50 iterations of {}".format(operator.__name__))
            # For progress on solving this, see #11
            return individual, log_args

    def mate(self, individual1: Individual, individual2: Individual, *args, **kwargs):
        def mate_with_log():
            new_individual1, new_individual2 = individual1.copy_as_new(), individual2.copy_as_new()
            self._mate(new_individual1, new_individual2, *args, **kwargs)
            log_args = [TOKENS.CROSSOVER, new_individual1._id, individual1._id, individual2._id]
            return new_individual1, log_args

        individual, log_args = self.try_until_new(mate_with_log)
        log_event(log, *log_args)
        return individual

    def mutate(self, individual: Individual, *args, **kwargs):
        def mutate_with_log():
            new_individual = individual.copy_as_new()
            mutator = self._mutate(new_individual, *args, **kwargs)
            log_args = [TOKENS.MUTATION, new_individual._id, individual._id, mutator.__name__]
            return new_individual, log_args

        individual, log_args = self.try_until_new(mutate_with_log)
        log_event(log, *log_args)
        return individual

    def individual(self, *args, **kwargs):
        expression = self._create_new(*args, **kwargs)
        compile_ = self._safe_compile if self._safe_compile is not None else self._compile
        return Individual(expression, to_pipeline=compile_)

    def create(self, *args, **kwargs):
        return self._create_from_population(self, *args, **kwargs)

    def eliminate(self, *args, **kwargs):
        return self._eliminate(*args, **kwargs)
