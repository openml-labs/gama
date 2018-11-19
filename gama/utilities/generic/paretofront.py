class ParetoFront(object):
    """  A Pareto front is a list of tuples in which no one tuple is dominated by another. """

    def __init__(self, start_list=None, get_values_fn=None):
        """
        :param start_list: list or None (default: None). List of items of which to calculate the Pareto front.
        :param get_values_fn: (default: None)
            function that takes an item and returns a tuple of values such that each should be maximized.
            If left None, it is assumed that items are already tuples of which each value should be maximized.
        """
        if get_values_fn:
            self._get_values_fn = get_values_fn
        else:
            self._get_values_fn = lambda x: x

        self._front = []
        if start_list:
            for item in start_list:
                self.update(item)

        self._iterator_index = 0

    def update(self, new_item):
        """ Updates the Pareto front with new_item if it dominates any current item in the Pareto front.

        :param new_item: Item to be evaluated for submission to the Pareto front. Arity must match that of items
          already in the Pareto front.
        :return: True if the Pareto front is updated, False otherwise.
        """
        if not self._front:
            self._front.append(new_item)
            return True

        values = self._get_values_fn(new_item)
        old_arity = len(self._get_values_fn(self._front[0]))
        if old_arity != len(values):
            raise ValueError("Arity of added tuple must match that of the ones in the Pareto front. Front tuples have "
                             "arity {} and new tuple has arity {}.".format(len(self._front[0]), len(values)))

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

    def clear(self):
        self._front = []

    def __iter__(self):
        self._iterator_index = 0
        return self

    def __next__(self):
        if self._iterator_index >= len(self._front):
            raise StopIteration

        self._iterator_index += 1
        return self._front[self._iterator_index - 1]

    def __len__(self):
        return len(self._front)

    def __getitem__(self, item):
        return self._front[item]

    def __str__(self):
        return str(self._front)

    def __repr__(self):
        # TODO: Add notification of non-standard `self._get_values_fn`.
        return 'ParetoFront({})'.format(str(self._front))
