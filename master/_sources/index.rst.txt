GAMA - General Automated Machine learning Assistant
===================================================

GAMA is a tool for Automated Machine Learning (AutoML).
All you need to do is supply the data, and GAMA will automatically try to find a good *machine learning pipeline*.
For the *machine learning pipeline* GAMA considers data preprocessing steps, various machine learning algorithms, and their possible hyperparameters configurations.
This takes away the knowledge and labour intensive work of selecting the right algorithms and tuning their hyperparameters yourself.
Using GAMA is as simple as using a scikit-learn estimator::

	from gama import GamaClassifier
	automl =  GamaClassifier()
	automl.fit(X_train, y_train)
	automl.predict(X_test)
	automl.predict_proba(X_test)

You can install GAMA from PyPI with pip::

    pip install gama

Or if you would like to include the graphical frontend::

    pip install gama[vis]

To get more basic information on GAMA and its AutoML functionality, read more in the :ref:`user_guide_index`.
If you want find out everything there is to know about GAMA, also visit the :ref:`advanced_guide_index`.
It describes visualization of optimization logs, changing the AutoML pipeline, and more.
If there are any questions you have that are not answered by the documentation, check the `issue page <https://github.com/openml-labs/gama/issues>`_.
If your question has not been answered there either, please open a new issue.

.. toctree::
   :includehidden:

   user_guide/index
   advanced_guide/index
   api/index
   benchmark
   releases
   contributing/index
   citing
