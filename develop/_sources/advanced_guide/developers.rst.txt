:orphan:


Developers Notes
----------------

.. _add_your_own:

Adding Your Own Search or Postprocessing
****************************************

.. note:: This is not set in stone. As more AutoML pipeline steps are added by more people,
    we expect to identify parts of the interface to be improved. We can't do this without your feedback!
    Feel free to get in touch, preferably in the form of a public discussion on a Github issue, and let us
    know what difficulties you encounter, or what works well!

This section contains information about implementing your own Search or Postprocessing procedures.
To keep interfaces uniform across the different search or postprocesing implementations, each should derive from their respective baseclass (`BaseSearch` and `BasePostProcessing`).
They each have their own processing method (`search` for `BaseSearch` and `post_process` for `BasePostProcessing`) which should be implemented.
We will show example implementations further down.

Your Search or Postprocessing algorithm may feature hyperparameters, care should be taken to provide good default values.
For some algorithms, hyperparameter default values are best specified based on characteristics of the dataset.
For instance, with big datasets it might be useful to perform (some of) the workload on a subset of the data.
We refer to these data-dependent non-static defaults as 'dynamic defaults'.
Both `BaseSearch` and `BasePostProcessing` feature a `dynamic_defaults` method which is called before `search` and `...`, respectively.
This allows you to overwrite default hyperparameter values based on the dataset properties.
The hyperparameter values with which your search or postprocessing will be called is determined in the following order:

    - User specified values are used if specified (e.g. `EnsemblePostProcessing(n=25)`)
    - Otherwise the values determined by `dynamic_defaults` are used
    - If neither are specified, the static default values are used.

Search
~~~~~~
To implement your own search procedure, create a class which derives from the base class:

.. autoclass:: gama.search_methods.base_search.BaseSearch
    :members:
    :noindex:

You can use existing search implementations as a reference.

`__init__`
^^^^^^^^^^
To allow us to identify which hyperparameters are set by the user, and which are defaults, the default values for each hyperparameter in the `__init__` method should be `None`.


In search methods, each evaluation of a machine learning pipeline is logged automatically.
Default data recorded includes:

    - a string representation of the pipeline
    - the scores of the pipeline according to the specified metrics
    - any errors that occurred during evaluation

It is possible to add additional fields to be recorded for each pipeline, as shown here.
The `extra_fields` of the `EvaluationLogger` expects a dictionary,
which maps the name of a field to the method which extracts the information that should be recorded.
In this case we are interested to know the parent of each evaluated pipeline, so we might later inspect a pipeline's "lineage".

`dynamic_defaults`
^^^^^^^^^^^^^^^^^^
Hyperparameters such as population size might make for good candidates for dynamic defaults.
However it is not obvious what the relationship should be.
For this reason, we choose not to work with dynamic defaults in this search strategy.
Perhaps in the future, when we have adequate data to model the relationship we can determine useful default values.

You can find an example usage of dynamic defaults in the Asynchronous Successive Halving Algorithm search.

`search`
^^^^^^^^
This method should execute the search for a good machine learning pipeline.
The search should always take into account the `start_candidates` in some form.
This allows the search start point to be set by the user or a warm start step.
In this evolutionary optimization, they form the initial population.

The search algorithm should update the `output` field of the `Search` object, and behave nicely when interrupted with a `TimeoutException`.
This allows GAMA to control when to shut down search (and continue with post processing).

PostProcessing
~~~~~~~~~~~~~~
PostProcessing follows a similar pattern, where the class should allow initialization with its hyperparameters,
an implementation of dynamic defaults (optional), and a `post_process` function.

.. autoclass:: gama.postprocessing.BasePostProcessing
    :members:
    :noindex:

Unlike the search methods, which are not required to have any hyperparameter, post processing is required to have a default value for `time_fraction`.
`time_fraction` is the fraction of the total time that should be reserved for the post processing method (as set on initialization through `max_total_time`).
For instance, when a post-processing object's `time_fraction` is `0.3` and GAMA is initiated with `max_total_time=3600`,
then `3600*0.3=1080` seconds are reserved for the post-processing phase.

.. note::
    While hard, it is important to provide an accurate estimate for `time_fraction`. If you reserve too much time,
    it means that the search procedure will have to be cut off unnecessarily early. If too little time is reserved,
    GAMA will interrupt the post-processing step and return control to the user.
    It is generally hard to know to how much to reserve, and is likely dependent on the dataset and number of evaluated
    pipelines in search. We would like to implement ways in which post-processing methods have access to these
    statistics and allow them to update their time estimate, so that less time is wasted on too long or too short
    post-processing phases.
