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


class HallOfFame(object):
    
    def __init__(self, filename):
        self._pareto_callbacks = []
        self._evaluation_callbacks = []
        self._front = ParetoFront(get_values_fn=lambda ind: ind.fitness.wvalues)
        self._filename = filename
        self._pop = []
        self._last_pop = []
        
    def update(self, pop):
        self._pop += pop
        self._last_pop = pop

        for ind in pop:
            updated = self._front.update(ind)
            if updated:
                self._update_pareto_front(ind)
        
        with open(self._filename, 'a') as fh:
            # print('-gen-')
            # print('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))
            fh.write('\n'.join([str((str(ind), ind.fitness.values[0])) for ind in pop]))

        self._update_evaluation(pop)

    def best_n(self, n):
        best_pipelines = sorted(self._pop, key=lambda x: (-x.fitness.values[0], str(x)))
        return best_pipelines[:n]

    def _update_evaluation(self, pop):
        for callback in self._evaluation_callbacks:
            callback(pop)

    def on_evaluations_received(self, fn):
        """ Register a callback function that is called when new evaluations are received.

        :param fn: Function to call when evaluations are received. Expected signature is: list: ind -> None
        """
        self._evaluation_callbacks.append(fn)

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
