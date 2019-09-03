Post-Processing
---------------

.. module:: gama.postprocessing

NoPostProcessing
****************

.. autoclass:: NoPostProcessing

BestFitPostProcessing
*********************

.. autoclass:: BestFitPostProcessing

EnsemblePostProcessing
**********************

.. autoclass:: EnsemblePostProcessing

Developer Note
**************

Each post processing method consists of two parts:

    - a class that is used to store its configuration and determine default values
    - a function which implements the search algorithm

Some algorithms' hyperparameters' default values are best specified based on characteristics of the dataset,
we refer to these non-static defaults as 'dynamic defaults'.
To keep these two functions together explicitly and keep naming consistent between different post-processing methods,
we bundle them in a class derived from `BasePostProcessing`:

.. autoclass:: BasePostProcessing

Through the dynamic defaults function, a post-processing method can retrieve any necessary information from GAMA.
Of particular interest may be the directory which caches predictions of each evaluated pipeline.
Through the time_fraction hyperparameter each method should also indicate what fraction of total time to reserve
for the post-processing phase. For instance, when a post-processing object's `time_fraction` is `0.3` and GAMA
is initiated with `max_total_time=3600`, then `3600*0.3=1080` seconds are reserved for the post-processing phase.
Refer to implemented post-processing techniques for examples.

.. note::
    While hard, it is important to provide an accurate estimate for `time_fraction`. If you reserve too much time,
    it means that the search procedure will have to be cut off unnecessarily early. If too little time is reserved,
    GAMA will interrupt the post-processing step and return control to the user.
    It is generally hard to know to how much to reserve, and is likely dependent on the dataset and number of evaluated
    pipelines in search. We would like to implement ways in which post-processing methods have access to these
    statistics and allow them to update their time estimate, so that less time is wasted on too long or too short
    post-processing phases.
