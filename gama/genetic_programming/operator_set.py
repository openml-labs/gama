import logging

from gama.utilities.logging_utilities import log_parseable_event, TOKENS
from .components import Individual

log = logging.getLogger(__name__)


class OperatorSet:
    """ Provides a thin layer for ea operators for logging and safety. """

    def __init__(self, mutate, mate, create_from_population, create_new, compile_, eliminate, max_retry=50):
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
        self._eliminate = eliminate
        self._max_retry = max_retry

        self._seen_individuals = {}

    def try_until_new(self, operator, *args, **kwargs):
        for _ in range(self._max_retry):
            individual, log_args = operator(*args, **kwargs)
            if str(individual.main_node) not in self._seen_individuals:
                self._seen_individuals[str(individual.main_node)] = individual
                return individual, log_args
        else:
            log.warning("Could not create a new individual from 50 iterations of {}".format(operator.__name__))
            return individual, log_args  # return as if new.. TODO: guarantee actual new (even if not through operator).

    def mate(self, individual1: Individual, individual2: Individual, *args, **kwargs):
        def mate_with_log():
            new_individual1, new_individual2 = individual1.copy_as_new(), individual2.copy_as_new()
            self._mate(new_individual1, new_individual2, *args, **kwargs)
            log_args = [TOKENS.CROSSOVER, new_individual1._id, individual1._id, individual2._id]
            return new_individual1, log_args

        individual, log_args = self.try_until_new(mate_with_log)
        log_parseable_event(log, *log_args)
        return individual

    def mutate(self, individual: Individual, *args, **kwargs):
        def mutate_with_log():
            new_individual = individual.copy_as_new()
            mutator = self._mutate(new_individual, *args, **kwargs)
            log_args = [TOKENS.MUTATION, new_individual._id, individual._id, mutator.__name__]
            return new_individual, log_args

        individual, log_args = self.try_until_new(mutate_with_log)
        log_parseable_event(log, *log_args)
        return individual

    def individual(self, *args, **kwargs):
        return self._create_new(*args, **kwargs)

    def create(self, *args, **kwargs):
        return self._create_from_population(self, *args, **kwargs)

    def compile(self, *args, **kwargs):
        return self._compile(*args, **kwargs)

    def eliminate(self, *args, **kwargs):
        return self._eliminate(*args, **kwargs)
