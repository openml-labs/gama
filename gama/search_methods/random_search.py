import logging

from gama.search_methods import _check_base_search_hyperparameters
from gama.utilities.generic.async_executor import AsyncExecutor

log = logging.getLogger(__name__)


def random_search(toolbox, output, start_candidates, restart_callback=None, max_n_evaluations=10000, n_jobs=1):
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
