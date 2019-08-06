import logging
from typing import List

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.genetic_programming.operator_set import OperatorSet
from gama.search_methods import _check_base_search_hyperparameters
from gama.search_methods.base_search import BaseSearch
from gama.utilities.generic.async_executor import AsyncExecutor

log = logging.getLogger(__name__)


class RandomSearch(BaseSearch):
    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time: int):
        pass

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        random_search(operations, self.output, start_candidates)


def random_search(toolbox, output, start_candidates):
    _check_base_search_hyperparameters(toolbox, output, start_candidates)

    futures = set()
    with AsyncExecutor() as async_:
        for individual in start_candidates:
            futures.add(async_.submit(toolbox.evaluate, individual))

        while True:
            done, not_done = toolbox.wait_first_complete(futures)
            for future in done:
                output.append(future.result())
                futures.add(async_.submit(toolbox.evaluate, toolbox.individual()))
