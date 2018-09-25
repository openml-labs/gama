.. gama documentation master file, created by
   sphinx-quickstart on Sun Aug 19 19:44:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. automodule:: gama
   :members:
   
GAMA - Genetic Automated Machine learning Assistant
================================

GAMA is a tool for Automated Machine Learning (AutoML).
All you need to do is supply the data, and GAMA will automatically try to find a good *machine learning pipeline*.
For the *machine learning pipeline* GAMA considers data preprocessing steps, various machine learning algorithms, and their possible hyperparameters configurations.
This takes away the knowledge and labour intensive work of selecting the right algorithms and tuning their hyperparameters yourself.
Using GAMA is as simple as using::

	from gama import GamaClassifier
	automl =  GamaClassifier()
	automl.fit(X_train, y_train)
	automl.predict(X_test)
	automl.predict_proba(X_test)

.. toctree::
   :maxdepth: 2
   
   getting_started
   configuration
   api/gamaclassifier
   generated/modules
   
   :caption: Contents:

Citing

Contributing

License


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
