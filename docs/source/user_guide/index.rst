..  _user_guide_index:

User Guide
===========
GAMA is an AutoML tool which aims to automatically find the right machine learning algorithms to create the best possible data for your model.
This page gives an introduction to basic components and concepts of GAMA.

GAMA performs a search over *machine learning pipelines*.
An example of a machine learning pipeline would be to first perform data normalization and then use a nearest neighbor classifier to make a prediction on the normalized data.
More formally, a *machine learning pipeline* is a sequence of one or more *components*.
A *component* is an algorithm which performs either data transformation *or* a prediction.
This means that components can be preprocessing algorithms such as PCA or standard scaling, or a predictor such as a decision tree or support vector machine.
A machine learning pipeline then consists of zero or more preprocessing components followed by a predictor component.

Given some data, GAMA will start a search to try and find the best possible machine learning pipelines for it.
After the search, the best model found can be used to make predictions.
Alternatively, GAMA can combine several models into an *ensemble* to take into account more than one model when making predictions.
For ease of use, GAMA provides a `fit`, `predict` and `predict_proba` function akin to scikit-learn.

.. include:: installation.rst
    :start-line: 1

-----

.. include:: examples.rst
    :start-line: 1

-----

.. include:: simple_features.rst
    :start-line: 1

-----

.. include:: dashboard.rst
    :start-line: 1

-----

.. include:: hyperparameters.rst
    :start-line: 1

-----

.. include:: related_packages.rst
    :start-line: 1
