class ParetoFront(object):
    """  """

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
