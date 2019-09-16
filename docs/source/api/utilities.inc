Utilities
---------

Generic
*******

Collection of generic components.

Pareto Front
~~~~~~~~~~~~

.. autoclass:: gama.utilities.generic.paretofront.ParetoFront

Stopwatch
~~~~~~~~~

.. autoclass:: gama.utilities.generic.stopwatch.Stopwatch

Timekeeper
~~~~~~~~~~

.. autoclass:: gama.utilities.generic.timekeeper.TimeKeeper

AsyncExecutor
~~~~~~~~~~~~~

.. warning::
    This component in particular has its own issues and will be replaced.
    But I have yet to find a satisfying alternative to 'hacking' the built-in ProcessPoolExecutor.

.. autoclass:: gama.utilities.generic.async_executor.AsyncExecutor
