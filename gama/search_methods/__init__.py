"""
This submodule contains the different search methods available in GAMA.
All search methods share a set of common hyperparameters, in addition to ones unique to each method.
The shared hyperparameters are:
 - toolbox
 - output: List[Individual]
    A list to be filled with all individuals still in consideration.
 - start_population: List[Individual]
    A list of individuals to be considered before all others.

Additionally, each search method should expect to be interrupted by a stopit.TimeoutException at any time.
Preferably the `output` list is up-to-date at that time and no further handling is required (see Random Search).
Alternatively, the exception can be caught so that the search algorithm can shut down elegantly when this happens,
and the final list of individuals may be returned (see ASHA).
"""
from typing import List

from gama.genetic_programming.components import Individual


def _check_base_search_hyperparameters(
        toolbox,
        output: List[Individual],
        start_candidates: List[Individual]
) -> None:
    """ Checks that search hyperparameters are valid.

    :param toolbox:
    :param output:
    :param start_candidates:
    :return:
    """
    if not isinstance(start_candidates, list):
        raise TypeError(f"'start_population' must be a list but was {type(start_candidates)}")
    if not all(isinstance(x, Individual) for x in start_candidates):
        raise TypeError(f"Each element in 'start_population' must be Individual.")


#__all__ = [asha, async_ea, random_search, _check_base_search_hyperparameters]
