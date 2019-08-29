:orphan:

Events
------

It is also possible to programmatically receive updates of the optimization process through the events::

    from gama import GamaClassifier

    def process_individual(individual):
        print('{} was evaluated. Fitness is {}.'.format(individual, individual.fitness.values))

    automl = GamaClassifier()
    automl.evaluation_completed(process_individual)
    automl.fit(X, y)

The function passed to ``evaluation_completed`` should take a ``gama.genetic_programming.components.individual.Individual``
as single argument.
Any exceptions raised but not handled in the callback will be ignored but logged at ``logging.WARNING`` level.
During the callback a ``stopit.utils.TimeoutException`` may be raised.
This signal normally indicates to GAMA to move on to the next step in the AutoML pipeline.
If caught by the callback, GAMA may exceed its allotted time.
For this reason, it is advised to keep callbacks short after catching a ``stopit.utils.TimeoutException``.
If the ``stopit.utils.TimeoutException`` is not caught, GAMA will correctly terminate its step in the AutoML pipeline
and continue as normal.
