"""
This submodule contains the different search methods available in GAMA.

.. note::
    You will notice an inconsistency between the signature default types and values,
    and the one in the description. The signature has `Optional` values which default
    to `None`.
    If left at `None`, they will default to the value specified in the description,
    which may be a value based on data characteristics.

"""

from gama.search_methods.asha import AsynchronousSuccessiveHalving
from gama.search_methods.async_ea import AsyncEA
from gama.search_methods.random_search import RandomSearch
from gama.search_methods.base_search import _check_base_search_hyperparameters


__all__ = ["AsynchronousSuccessiveHalving", "AsyncEA", "RandomSearch"]
