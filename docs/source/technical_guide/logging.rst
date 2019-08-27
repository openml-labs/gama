:orphan:

.. default-role:: code


Logging
-------

GAMA makes use of the default Python `logging <https://docs.python.org/3.5/library/logging.html>`_ module.
This means logs can be captured at different levels, and handled by one of several StreamHandlers.
In addition to the Python built-in log levels GAMA introduces one level below `logging.DEBUG`, explicitly for log
messages that are meant to be parsed by a program later.

The most common logging use cases are to write a comprehensive log to file, as well as print important messages to `stdout`.
Writing log messages to `stdout` is directly supported by GAMA through the `verbosity` hyperparameter
(which defaults to `logging.WARNING`).

Logging all log messages to file is not entirely supported directly.
Instead, GAMA provides a `keep_analysis_log` hyperparameter.
When set (by providing the name of the file to write to), it will only write *most* information to file.
In particular it will write any log messages at `logging.WARNING`, `logging.CRITICAL` and `logging.ERROR`, as
well as all messages at the `MACHINE_LOG_LEVEL` - the log level introduced by GAMA.
The `MACHINE_LOG_LEVEL` write messages in a structured way, so that they may easily be parsed by a program later.
By default GAMA writes to 'gama.log'.

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
    automl = GamaClassifier(max_total_time=180, verbosity=logging.DEBUG, keep_analysis_log=None)

Running the above script will create the 'logfile.txt' file with all log messages that could also be seen in the console.
An overview the log levels:

 - `MACHINE_LOG_LEVEL (5)`: Messages mostly meant to be parsed by a program.
 - `DEBUG`: Messages for developers.
 - `INFO`: General information about the optimization process.
 - `WARNING`: Serious errors that do not prohibit GAMA from running to completion (but results could be suboptimal).
 - `ERROR`: Errors which prevent GAMA from running to completion.


Generated Files
***************

GAMA will create some files during the optimization process. Here is an overview of produced files:

**Folder `'{DATE}_{STARTTIME}_XXXX_GAMA'`**: (XXXX are random alphanumeric characters)
This folder is used to save results from evaluations during the optimization process.
The files in this folders are needed to create an ensemble of pipelines in the post-processing phase of GAMA.
You can specify the name of this folder with the `cache_dir` hyperparameter when initializing a GAMA object.
By default, this folders gets deleted after the `fit` call is done.
In order to preserve this folder (e.g. to later construct different ensembles or later analysis), specify `keep_cache=True` when calling `fit`.

**File: gama.log**: This file contains all information about the optimization process, and by default is not removed.
As described in `Log Visualization`_ this file can be used to generate visualizations about the optimization process.
If you wish to have the file be automatically deleted, specify `keep_analysis_log=False` when initializing a GAMA object, as per the `Examples`_.