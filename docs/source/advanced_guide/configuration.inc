:orphan:

.. _search_space_configuration:


GAMA Search Space Configuration
-------------------------------

By default GAMA will build pipelines out of scikit-learn algorithms, both for preprocessing and learning models.
It is possible to modify this search space, changing the algorithms or hyperparameter ranges to consider.

The search space is determined by the `search_space` dictionary passed upon initialization.
The defaults are found in
`classification.py <https://github.com/openml-labs/gama/tree/master/gama/configuration/classification.py>`_ and
`regression.py <https://github.com/openml-labs/gama/tree/master/gama/configuration/regression.py>`_
for the GamaClassifier and GamaRegressor, respectively.

A sample of algorithms that GAMA uses by default:

 - `logistic regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
 - `random forest classifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
 - `naive bayes <https://scikit-learn.org/stable/modules/naive_bayes.html>`_
 - `support vector machines <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_
 - `PCA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
 - `normalization <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html>`_
 - `ICA <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html>`_

The search space configuration is defined in a python dictionary.
For reference, a minimal example search space configuration can look like this::

    from sklearn.naive_bayes import BernoulliNB
    search_space = {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        BernoulliNB: {
            'alpha': [],
            'fit_prior': [True, False]
        }
    }


At the top level, allowed key types are:

* ``string``, with a list as value.

 It specifies the name of a hyperparameter with its possible values.
 By defining a hyperparameter at the top level, you can reference it as hyperparameter for any specific algorithm.
 To do so, identify it with the same name and set its possible values to an empty list (see `alpha` in the example).
 The benefit of doing is that multiple algorithms can share a hyperparameter space that is defined only once.
 Additionally, in evolution this makes it possible to know which hyperparameter values can be crossed over between
 different algorithms.

* ``class``, with a dictionary as value.

 The key specifies the algorithm, calling it should instantiate the algorithm.
 The dictionary specifies the hyperparameters by name and their possible values as list.
 All hyperparameters specified should be taken as arguments for the algorithm's initialization.
 A hyperparameter specified at the top level of the dictionary can share a name with a hyperparameter of the algorithm.
 To use the values provided by the shared hyperparameter, set the possible values to an empty list.
 If a list of values is provided instead, it will not use the shared hyperparameter values.
