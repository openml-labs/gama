:orphan:

Important Hyperparameters
-------------------------

There are a lot of hyperparameters exposed in GAMA.
In this section, you will find some hyperparameters you might want to set even if you otherwise use defaults.
For more complete documentation on all hyperparameters, see :ref:`API documentation <api_doc>`.

Optimization
************
Perhaps the most important hyperparameters are the ones that specify what to optimize for, these are:

``scoring``: ``string`` (default='log_loss' for classification and 'mean_squared_error' for regression)
    Sets the metric to optimize for. Make sure to optimize towards the metric that reflects well what is important to you.
    Valid options include `roc_auc`, `accuracy` and `log_loss` for classification, and `mean_squared_error` and `r2` for regression.
    For more options see :ref:`API documentation <api_doc>`.

``regularize_length``: ``bool`` (default=True)
    If True, in addition to optimizing towards the metric set in ``scoring``, also guide the search towards shorter pipelines.
    This setting currently has no effect for non-default search methods.


Example::

    GamaClassifier(scoring='roc_auc', regularize_length=False)

Resources
*********

``n_jobs``: ``int`` (default=-1)
    Determines how many processes can be run in parallel during `fit`.
    This has the most influence over how many machine learning pipelines can be evaluated.
    If it is set to -1, which is the default, all cores are used.
    Changing it to use a set amount of (fewer) cores will decrease the amount of pipelines evaluated,
    but is needed if you do not want GAMA to use all resources.

``max_total_time``: ``int`` (default=3600)
    The maximum time in seconds that GAMA should aim to use to construct a model from the data.
    By default GAMA uses one hour. For large datasets, more time may be needed to get useful results.

``max_eval_time``: ``int`` (default=300)
    The maximum time in seconds that GAMA is allowed to use to evaluate a single machine learning pipeline.
    The default is set to five minutes. For large datasets, more time may be needed to get useful results.

Example::

    GamaClassifier(n_jobs=2, max_total_time=7200, max_eval_time=600)
