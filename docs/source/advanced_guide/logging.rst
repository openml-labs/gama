:orphan:

.. default-role:: code

.. _logging-section:

Logging
-------

GAMA makes use of the default Python `logging <https://docs.python.org/3.5/library/logging.html>`_ module.
This means logs can be captured at different levels, and handled by one of several StreamHandlers.

The most common logging use cases are to write a comprehensive log to file, as well as print important messages to `stdout`.
Writing log messages to `stdout` is directly supported by GAMA through the `verbosity` hyperparameter
(which defaults to `logging.WARNING`).

By default GAMA will also save several different logs.
This can be turned off by the `store` hyperparameter.
The `store` hyperparameter allows you to store the logs, as well as models and predictions.
By default logs are kept (which includes evaluation data), but models and predictions are discarded.

The `output_directory` hyperparameter determines where this data is stored, by default a unique name is generated.
In the output directory you will find three files and a subdirectory:

 - 'evaluations.log': a csv file (with ';' as separator) in which each evaluation is stored.
 - 'gama.log': A loosely structured file with general (human readable) information of the GAMA run.
 - 'resources.log': A record of the memory usage for each of GAMA's processes over time.
 - cache directory: contains evaluated models and predictions, only if `store` is 'all' or 'models'

If you want other behavior, the logging module offers you great flexibility on making your own variations.
The following script writes any log messages of `logging.DEBUG` or up to both file and console::

    import logging
    import sys
    from gama import GamaClassifier

    gama_log = logging.getLogger('gama')
    gama_log.setLevel(logging.DEBUG)

    fh_log = logging.FileHandler('logfile.txt')
    fh_log.setLevel(logging.DEBUG)
    gama_log.addHandler(fh_log)

    # The verbosity hyperparameter sets up an StreamHandler to `stdout`.
    automl = GamaClassifier(max_total_time=180, verbosity=logging.DEBUG, store="nothing")

Running the above script will create the 'logfile.txt' file with all log messages that could also be seen in the console.
An overview the log levels:

 - `DEBUG`: Messages for developers.
 - `INFO`: General information about the optimization process.
 - `WARNING`: Serious errors that do not prohibit GAMA from running to completion (but results could be suboptimal).
 - `ERROR`: Errors which prevent GAMA from running to completion.

As described in :ref:`dashboard-section` the files in the output directory can be used to generate visualizations about the optimization process.
