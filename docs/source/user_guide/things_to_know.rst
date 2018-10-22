Things To Know
--------------
GAMA is in its early stages and very much work in progress.
Some things won't change, such as providing a scikit-learn like interface as well as ARFF support.
However, many other things such as how search space configuration or even the search itself is done, can *and probably will* still change.

Important Hyperparameters
*************************

GAMA has a lot of hyperparameters to tweak.
Finding the best defaults is always a work in progress, so if you run into issues, try tuning some of them.

Here is a selection that might be of particular interest that are accessed on initialization:

**objectives**: states towards which metrics to optimize.
Supplying a binary tuple allows for multi-objective optimization.
Make sure to optimize towards the metric that reflects well what is important to you.

**n_jobs**: determines how many processes can be run in parallel during `fit`.
By default only one core is used, so if you have more cores available, this has the most influence over how many
machine learning pipelines can be evaluated.

**max_total_time**: the maximum time in seconds that GAMA should aim to use to construct a model from the data.
By default GAMA uses one hour. For large datasets, more time may be needed to get useful results.

**max_eval_time**: the maximum time in seconds that GAMA is allowed to use to evaluate a single machine learning pipeline.
By default GAMA uses five minutes. For large datasets, more time may be needed to get useful results.

The `fit` function can also be supplied with some optional hyperparameters:

**auto_ensemble_n**: This hyperparameter specifies the number of models to include in the final ensemble.
The final ensemble may contain duplicates as a form of assigning weights.

**keep_cache**: During the optimization process, each model evaluated is stored alongside its predictions.
This is needed for automatic ensemble construction.
Normally, this cache is deleted automatically, but should you wish to keep it, you can specify it here.


Logging
*******

GAMA makes use of the default Python `logging <https://docs.python.org/3.5/library/logging.html>`_ module.
The log can be captured at different levels, and handled by one of several StreamHandlers.

The most common use cases would be to write a comprehensive log to file, as well as print important messages to `stdout`.
Both of these cases are directly supported by GAMA through the `verbosity` and `keep_analysis_log` hyperparameters.
The level of `verbosity` defines what level messages should be written to `stdout`, and is by default set to `logging.WARNING`.
The `keep_analysis_log` sets whether or not to write *all* log output to file (`logging.DEBUG` level).

However, the logging module offers you great flexibility on making your own variations.
The following script manually sets up GAMA to print to stdout (and ignores the built-in)::

    import logging
    import sys
    from gama import GamaClassifier

    gama_log = logging.getLogger('gama')
    gama_log.setLevel(logging.INFO)

    stdout_streamhandler = logging.StreamHandler(sys.stdout)
    stdout_streamhandler.setLevel(logging.INFO)
    gama_log.addHandler(stdout_streamhandler)

    automl = GamaClassifier(max_total_time=180, verbosity=logging.ERROR)

Running the above script display the GAMA version used, and the hyperparameter values on initialization::

    Using GAMA version 0.1.0.
    GamaClassifier(cache_dir=None,verbosity=None,n_jobs=1,max_eval_time=300,max_total_time=180,population_size=50,random_state=None,optimize_strategy=(1, -1),objectives=('neg_log_loss', 'size'))

Actual values or hyperparameter names may vary depending on the version of GAMA you are using.
Make sure to set *both* the log level of the log and the stream handler.

Log Visualization
*****************

When using the default hyperparameters, GAMA will produce a log file called `gama.log`.
This log file is structured so that important events are easy to parse.
For an example and easy way to visualize information from the log file, take a look at [this notebook](https://github.com/PGijsbers/gama/blob/master/notebooks/GAMA%20Log%20Parser.ipynb).

Events and Observers
********************

It is also possible to programmatically receive updates of the optimization process through the events::

    from gama import GamaClassifier

    def process_individual(individual):
        print('{} was evaluated. Fitness is {}.'.format(individual, individual.fitness.values))

    automl = GamaClassifier()
    automl.evaluation_completed(process_individual)
    automl.fit(X, y)

This can be used to create useful observers, such as one that keeps track of the Pareto front or visualizes progress.