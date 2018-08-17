import logging

log = logging.getLogger(__name__)


class ParetoFront(object):

    def __init__(self, get_values_fn=None):
        """
        :param get_values_fn: (default: None)
            function that takes an item and returns a tuple of values such that each should be maximized.
            If left None, it is assumed that items are already tuples of which each value should be maximized.
        """
        if get_values_fn:
            self._get_values_fn = get_values_fn
        else:
            self._get_values_fn = lambda x: x

        self._front = []

    def update(self, new_item):
        """ Updates the Pareto-front with new_item if it dominates any current item in the Pareto-front.

        :param new_item: Item to be evaluated for submission to the Pareto-front.
        :return: True if the pareto-front is updated, False otherwise.
        """
        if not self._front:
            self._front.append(new_item)
            return True

        values = self._get_values_fn(new_item)
        to_remove = []

        # Check for each point whether new_item dominates it, it gets dominated, or neither.
        for old_item in self._front:
            old_values = self._get_values_fn(old_item)
            if all(v1 <= v2 for v1, v2 in zip(values, old_values)):
                # old_item dominates this new_item
                return False
            elif all(v1 >= v2 for v1, v2 in zip(values, old_values)):
                # new_item dominates this old_item
                to_remove.append(old_item)
            # else: Neither dominates nor gets dominated by old_item, which means new_item is still a candidate

        # new_item was not dominated by any Pareto-front item, which means it gets added.
        # so old items that new_item dominates must be removed.
        for item in to_remove:
            self._front.remove(item)
        self._front.append(new_item)
        return True


class Observer(object):
    
    def __init__(self, id_):
        self._overall_pareto_front = ParetoFront(get_values_fn=lambda ind: ind.fitness.wvalues)
        self._current_pareto_front = ParetoFront(get_values_fn=lambda ind: ind.fitness.wvalues)

        self._pareto_callbacks = []

        self._individuals = []
        self._individuals_since_last_pareto_update = 0

        self._evaluation_filename = str(id_)+'_evaluations.csv'

    def _record_individual(self, ind):
        with open(self._evaluation_filename, 'a') as fh:
            to_record = [str(ind.fitness.time),
                         str(ind.fitness.wvalues[0]),
                         str(ind.fitness.wvalues[1]),
                         str(ind)]
            fh.write(';'.join(to_record) + '\n')

    def update(self, ind):
        log.debug("Evaluation;{:.4f};{};{}".format(ind.fitness.time, ind.fitness.wvalues, ind))
        self._individuals.append(ind)
        self._record_individual(ind)

        updated = self._current_pareto_front.update(ind)
        if updated:
            self._individuals_since_last_pareto_update = 0
            log.info("Current pareto-front updated with individual with wvalues {}.".format(ind.fitness.wvalues))
        else:
            self._individuals_since_last_pareto_update += 1

        updated = self._overall_pareto_front.update(ind)
        if updated:
            self._update_pareto_front(ind)
            log.info("Overall pareto-front updated with individual with wvalues {}.".format(ind.fitness.wvalues))

    def reset_current_pareto_front(self):
        self._current_pareto_front._front = []
        self._individuals_since_last_pareto_update = 0

    def best_n(self, n):
        """ Return the best n individuals observed based on the first optimization criterion.

        :param n: the number of individuals to return
        :return: a list of up to n individuals for which the score on the first criterion is the best.
                returns less than n individuals if less than n have been evaluated.
        """
        best_pipelines = sorted(self._individuals, key=lambda x: (-x.fitness.values[0], str(x)))
        return best_pipelines[:n]

    def _update_pareto_front(self, ind):
        for callback in self._pareto_callbacks:
            callback(ind)

    def on_pareto_updated(self, fn):
        """ Register a callback function that is called when the Pareto-front is updated.

        :param fn: Function to call when the pareto front is updated. Expected signature is: ind -> None
        """
        self._pareto_callbacks.append(fn)

    def callback_on_improvement(self, fn, criterion=None):
        """ Register a callback function for when a certain criterion is improved upon in the pareto front.

        :param fn:
        :param criterion:
        :return:
        """
        raise NotImplemented()
