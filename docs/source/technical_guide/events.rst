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

This can be used to create useful observers, such as one that keeps track of the Pareto front or visualizes progress.
