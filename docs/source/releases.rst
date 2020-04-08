Release Notes
=============


Version 20.1.1
--------------
Features:
 # 70: Exported code is now auto-formatted with Black.

Maintenance:
 - Docs are automatically built and deployed on a commit to master and develop.
 - Pre-commit configuration added to check formatting, style and type hints on commit.
 - Black codestyle adapted, most drastic change is line length from 120 to 88.
 - Coverage increased by removing unused code, updating configuration, adding tests.

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