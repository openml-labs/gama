from collections import Sequence
from typing import Tuple, List, Optional, Callable, Any


class ParetoFront(Sequence):
    """ A list of tuples in which no one tuple is dominated by another. """

    def __init__(
        self,
        start_list: Optional[List[Any]] = None,
        get_values_fn: Optional[Callable[[Any], Tuple[Any, ...]]] = None,
    ):
        """
        Parameters
        ----------
        start_list: list, optional (default=None).
            List of items of which to calculate the Pareto front.
        get_values_fn: Callable, optional (default=None)
            Function that takes an item and returns a tuple of values,
            such that each should be maximized.
            If left None, it is assumed that items are already such tuples.
        """
        self._get_values_fn = get_values_fn

        self._front: List[Any] = []
        if start_list:
            for item in start_list:
                self.update(item)

        self._iterator_index = 0

    def _get_item_value(self, item):
        if self._get_values_fn is not None:
            return self._get_values_fn(item)
        else:
            return item

    def update(self, new_item: Any):
        """ Update the Pareto front with new_item if it qualifies.

        Parameters
        ----------
        new_item: Any
            Item to be evaluated for submission to the Pareto front.
            Either a Tuple that matches the arity of items already in the Pareto front,
            or an object from which such a Tuple can be extracted
            with the `get_values_fn` provided on `__init__`.

        Returns
        -------
        bool
            True if the Pareto front is updated, False otherwise.
        """
        if not self._front:
            self._front.append(new_item)
            return True

        values = self._get_item_value(new_item)
        old_arity = len(self._get_item_value(self._front[0]))
        if old_arity != len(values):
            raise ValueError(
                "Arity of added tuple must match that of the ones in the Pareto front. "
                f"Front tuples have arity {len(self._front[0])} and "
                f"new tuple has arity {len(values)}."
            )

        to_remove = []

        for old_item in self._front:
            old_values = self._get_item_value(old_item)
            if all(v1 <= v2 for v1, v2 in zip(values, old_values)):
                # old_item dominates this new_item, no update
                return False
            elif all(v1 >= v2 for v1, v2 in zip(values, old_values)):
                # new_item dominates this old_item
                to_remove.append(old_item)
            # else: Neither dominates nor gets dominated by old_item

        # new_item was not dominated by any Pareto-front item, update front:
        for item in to_remove:
            self._front.remove(item)
        self._front.append(new_item)
        return True

    def clear(self):
        """ Removes all items from the Pareto front."""
        self._front = []

    def __len__(self):
        return len(self._front)

    def __getitem__(self, item):
        return self._front[item]

    def __str__(self):
        return str(self._front)

    def __repr__(self):
        if self._get_values_fn is not None:
            fn_name = f", get_values_fn = '{self._get_values_fn.__name__}"
        else:
            fn_name = ""
        return f"ParetoFront({self._front}{fn_name})"
