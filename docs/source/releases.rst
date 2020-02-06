Release Notes
=============

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