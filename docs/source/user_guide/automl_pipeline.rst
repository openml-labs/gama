:orphan:

AutoML Pipeline
---------------

An AutoML system performs several operations in its search for a model, and each of them may have hyperparameters.
For example, on initialization search may be started randomly or warm-started based on a 'meta-model'.
An important choice is the search algorithm, which performs search over machine learning pipelines for your data.
Another choice would be how to construct a model after search, e.g. by training the best pipeline or constructing an ensemble.
Much like data processing algorithms can form a *machine learning pipeline*,
we will refer to a configuration of these AutoML components as an *AutoML Pipeline*.
In GAMA we currently support flexibility in the AutoML pipeline in three stages: pre-processing, search and post-processing.

Pre-processing
**************


Search Algorithms
*****************


Post-processing
***************

The following post-processing steps are available:
 -


Configuring the pipeline
************************

By default 'prepend pipeline', 'Asynchronous EA' and 'FitBest' are chosen for pre-processing, search and post-processing, respectively.
However, it is easy to change this, or to change the hyperparameters with which each component is used.
For example, searching with 'Asynchronous Successive Halving' and creating an ensemble during post-processing::

    from gama import GamaClassifier
    from gama.search_methods import ASHA  #Asynchronous Successive Halving
    from gama.postprocessing import Ensemble

    custom_pipeline_gama = GamaClassifier(search=ASHA(), post_processing=Ensemble)

or instead using 'Asynchronous EA' but with custom hyperparameters::

    from gama import GamaClassifier
    from gama.search_methods import AsyncEA

    custom_pipeline_gama = GamaClassifier(search=AsyncEA(population_size=30))

