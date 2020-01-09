import logging
from typing import List, Callable

from gama.genetic_programming.components import Individual
from gama.utilities.generic.paretofront import ParetoFront

log = logging.getLogger(__name__)


class Observer(object):
    
    def __init__(self, id_, with_log=False):
        self._with_log = with_log

        self._multiple_pareto_fronts = False
        self._overall_pareto_front = ParetoFront(get_values_fn=lambda ind: ind.fitness.values)
        self._current_pareto_front = ParetoFront(get_values_fn=lambda ind: ind.fitness.values)

        self._pareto_callbacks = []

        self._individuals = []
        self._individuals_since_last_pareto_update = 0

        self._evaluation_filename = str(id_)+'_evaluations.csv'

    def _record_individual(self, ind):
        with open(self._evaluation_filename, 'a') as fh:
            to_record = [str(ind.fitness.time),
                         str(ind.fitness.values[0]),
                         str(ind.fitness.values[1]),
                         str(ind)]
            fh.write(';'.join(to_record) + '\n')

    def update(self, ind: Individual):
        log.debug("Evaluation;{:.4f};{};{}".format(ind.fitness.wallclock_time, ind.fitness.values, ind))
        self._individuals.append(ind)
        if self._with_log:
            self._record_individual(ind)

        updated = self._current_pareto_front.update(ind)
        if updated:
            self._individuals_since_last_pareto_update = 0
            log.info("Current Pareto front updated: {} scores {}".format(ind.short_name(), ind.fitness.values))
        else:
            self._individuals_since_last_pareto_update += 1

        updated = self._overall_pareto_front.update(ind)
        if updated and self._multiple_pareto_fronts:
            self._update_pareto_front(ind)
            log.info("Overall pareto-front updated with individual with wvalues {}.".format(ind.fitness.values))

    def reset_current_pareto_front(self):
        self._current_pareto_front.clear()
        self._individuals_since_last_pareto_update = 0
        self._multiple_pareto_fronts = True

    def best_n(self, n: int) -> List[Individual]:
        """ Return the best n individuals observed based on the first optimization criterion.

        Parameters
        ----------
        n: int
            the number of individuals to return

        Returns
        -------
        List[Individual]
            A list of up to n individuals for which the score on the first criterion is the best.
            Returns less than n individuals if less than n have been evaluated.
        """
        best_pipelines = sorted(self._individuals, key=lambda x: (-x.fitness.values[0], str(x)))
        return best_pipelines[:n]

    def _update_pareto_front(self, ind):
        for callback in self._pareto_callbacks:
            callback(ind)

    def on_pareto_updated(self, fn: Callable[[Individual], None]):
        """ Register a callback function that is called when the Pareto-front is updated.

        Parameters
        ----------
        fn: Callable[[Individual], None]
            Function to call when the pareto front is updated. Expected signature is: ind -> None
        """
        self._pareto_callbacks.append(fn)
