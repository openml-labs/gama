:orphan:


AutoML Pipeline
---------------

An AutoML system performs several operations in its search for a model, and each of them may have several options and hyperparameters.
An important decision is picking the search algorithm, which performs search over machine learning pipelines for your data.
Another choice would be how to construct a model after search, e.g. by training the best pipeline or constructing an ensemble.
Similarly to how data processing algorithms can form a *machine learning pipeline*,
we will refer to a configuration of these AutoML components as an *AutoML Pipeline*.
In GAMA we currently support flexibility in the AutoML pipeline in two stages: search and post-processing.
See :ref:`add_your_own` for more information on how to add your own.

Search Algorithms
*****************
The following search algorithms are available in GAMA:

* ``Random Search``: Randomly pick machine learning pipelines from the search space and evaluate them.
* ``Asynchronous Evolutionary Algorithm``: Evolve a population of machine learning pipelines, drawing new machine learning pipelines from the best of the population.
* ``Asynchronous Successive Halving Algorithm``: A bandit-based approach where many machine learning pipelines iteratively get evaluated and eliminated on bigger fractions of the data.

Post-processing
***************
The following post-processing steps are available:

- ``None``: no post-processing will be done. This means no final pipeline will be trained and `predict` and `predict_proba` will be unavailable. This can be interesting if you are only interested in the search procedure.
- ``FitBest``: fit the single best machine learning pipeline found during search.
- ``Ensemble``: create an ensemble out of evaluated machine learning pipelines. This requires more time but can lead to better results.


Configuring the AutoML pipeline
*******************************

By default 'prepend pipeline', 'Asynchronous EA' and 'FitBest' are chosen for pre-processing, search and post-processing, respectively.
However, it is easy to change this, or to change the hyperparameters with which each component is used.
For example, searching with 'Asynchronous Successive Halving' and creating an ensemble during post-processing::

    from gama import GamaClassifier
    from gama.search_methods import AsynchronousSuccessiveHalving
    from gama.postprocessing import EnsemblePostProcessing

    custom_pipeline_gama = GamaClassifier(search=AsynchronousSuccessiveHalving(), post_processing=EnsemblePostProcessing())

or using 'Asynchronous EA' but with custom hyperparameters::

    from gama import GamaClassifier
    from gama.search_methods import AsyncEA

    custom_pipeline_gama = GamaClassifier(search=AsyncEA(population_size=30))

