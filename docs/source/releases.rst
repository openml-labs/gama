Release Notes
=============

Version 21.0.1
--------------

News:
 - We have a logo! Check out our README, or `images\logos` for several renders.

Bugfixes:
 - Check if `max_pipeline_length` is compatible with the search space, i.e. if there are no preprocessing steps in the search space it is set to 1 and raises an error if already set to a value greater than 1.
 - `mut_replace_primitive` mutation is no longer considered if there are no suitable replacements for any primitive in the individual.
 - Setting `store` to `"all"`.
 - Setting `n_jobs` to `-1` once more correctly creates one subprocess per core.
 - No longer leave zombie processes when `fit` is called more than once per process.

Contributors (alphabetical order):
 - @Bilgecelik
 - @PGijsbers

Version 21.0.0
--------------

Features:
 - The ``warm_start`` parameter of ``fit()`` had a slight behavior change (see docs).
 - Fit can now be called more than once. It will use the same time constraint as the first fit call.

Bugfixes:
 - TargetEncoder is no longer used for Classification tasks, since there is a `bug <https://github.com/scikit-learn-contrib/category_encoders/issues/182>`_ that prevents it from working as intended.


Version 20.2.2
--------------

Maintenance:
 - Minor refactoring
 - Updated and expanded documentation

Bugfixes:
 - Avoid a cache-related error when using ASHA search.

Version 20.2.1
--------------
Changes:
 # 24: Changes to logging
    The structure of the log file(s) have changed.
    The goal is to make the log files easier to use, by making them easier to read and
    easier to extend write behavior.
    There will now be three log files, one which contains just evaluation data, one which contains progress data, and one which contains resource usage data.
    For more information see :ref:`logging-section` in the technical guide.


Features:
 # 66: csv files are now supported.
    Call `fit_arff` is now `fit_from_file` which accepts both arff and csv files.
    The CLI interface and Gama Dashboard also allow for csv files.
 # 92: You can specify a memory limit through `max_memory_mb` hyperparameter.
    GAMA does not guarantee it will not violate the constraint, but violations
    should be infrequent and minor. Feel free to open an issue if you experience a
    violation which does not minor.

Version 20.2.0
--------------
Features:
 # 70: Exported code is now auto-formatted with Black.
 - The Dashboard now has an Analysis tab that allows you to load old log files.
 - The Dashboard Home tab allows you to view the data set.
 - The Dashboard Home tab allows you to select a target column.
 - Pipelines and estimators are now cached on disk again.
 - KNN and PolynomialFeatures are now dynamically disabled based on dataset size.

Maintenance:
 - Docs are automatically built and deployed on a commit to master and develop.
 - Pre-commit configuration added to check formatting, style and type hints on commit.
 - Black codestyle adapted, most drastic change is line length from 120 to 88.
 - Coverage increased by removing unused code, updating configuration, adding tests.
 - Memory usage of all GAMA's processes is logged.

Version 20.1.0
--------------
Features:
 - Encoding of ARFF files may now be specified with the `encoding` parameter in {fit/predict/score}_arff calls.
 - Set `max_pipeline_length` on initialization to impose a maximum number of steps in your pipelines.

Bugfixes:
 - Reading ARFF markers (such as @data and @attribute) is now correctly case insensitive.

Maintenance:
 - Evaluation results are no longer saved to disk but kept in memory only.
   Consequently the `cache_dir` hyperparameter has been removed.

Changes:
 - Pipelines fitted during search are now used in the ensemble, instead of retraining the pipeline.
 - Ordinal Encoding and One Hot Encoding are now applied outside of 5-fold CV.
   This is for computational reasons, as all levels of a categorical variable are known this shouldn't make a difference.

Version 20.0.0
--------------
Features:
 #65 GAMA Command Line Interface:
    Allows users to start GAMA from the command line.
    Requires data to be formatted in ARFF.
 #69 Code export:
    Export Python code that sets up the machine learning pipeline found with AutoML.
 #71 GAMA Dashboard:
    First steps to providing a user interface for GAMA.
    It allows users to start GAMA AutoML through a webapp built with Dash,
    and monitor the performance of the search as it is executed.

Bugfixes:
 #68: Only add categorical encoding steps if (non-binary) categorical data is present.

Maintenance:
 #67: Selection now takes crowding distance into account (again).
 #68: `n_jobs` will now default to use half of available cores.
 #68: Updates given about the Pareto front now include the pipeline structure.
 #70: Versioning now YY.Minor.Micro


Version 19.11.2
---------------
Bugfixes:
 - `predict_proba_arff` now also accepts a `target_column` as expected from the previous update.

Version 19.11.1
---------------
Features:
 - `gama.__version__` can now be used to retrieve gama's version.
 - `fit_arff`, `score_arff` and `predict_arff` now accept a `target_column` parameter to specify the target.
   If left unset, the last column of the ARFF file is assumed to be the target column.

Bugfixes:
 - fit(x, y) may now be called with y as (N,1) array.
 - ensemble post-processing is now compatible with non-zero indexed class labels

Maintenance:
 - `__version__.py` is now the only place with hard-coded version.

Version 19.11.0
---------------
Accidentally released without updates.


Version 19.08.0
---------------
- Prototype dash app for visualizing GAMA logs.
- Easy switching between search algorithms

Version 0.1.0
-------------
First GAMA release.
