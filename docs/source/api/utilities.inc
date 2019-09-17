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

AsyncEvaluator
~~~~~~~~~~~~~~

.. warning::
    I'm sure there are better tools out there, but I have yet to find a minimal easy multi-processing tool.
    I tried using the built-in ProcessPoolExecutor, but it had short comings such as not being able to cancel
    jobs while they were running.

.. autoclass:: gama.utilities.generic.async_evaluator.AsyncEvaluator
