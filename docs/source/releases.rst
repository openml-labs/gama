Release Notes
=============

Version 19.11.0
---------------
Features:
 - `gama.__version__` can now be used to retrieve gama's version.
 - `fit_arff`, `score_arff` and `predict_arff` now accept a `target_column` parameter to specify the target.
   If left unset, the last column of the ARFF file is assumed to be the target column.

Bugfixes:
 - fit(x, y) may now be called with y as (N,1) array.

Maintenance:
 - `__version__.py` is now the only place with hard-coded version.


Version 19.08.0
---------------
- Prototype dash app for visualizing GAMA logs.
- Easy switching between search algorithms

Version 0.1.0
-------------
First GAMA release.