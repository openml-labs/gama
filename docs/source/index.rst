.. gama documentation master file, created by
   sphinx-quickstart on Sun Aug 19 19:44:17 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. automodule:: gama
   :members:
   
GAMA - Genetic Automated Machine learning Assistant
===================================================

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

You can install GAMA (and its dependencies) by cloning the repository and calling the setup script::

    git clone https://github.com/PGijsbers/gama.git
    cd gama
    python setup.py install

To get more basic information on GAMA and its AutoML functionality, read more in the :ref:`user_guide_index`.
If you want find out everything there is to know about GAMA, also visit the :ref:`technical_guide_index`.
It describes visualization of optimization logs, changing the AutoML pipeline, and more.
If there are any questions you have that are not answered by the documentation, check the `issue page <https://github.com/PGijsbers/GAMA/issues>`_.
If your question has not been answered there either, please open a new issue and label with the question label.

.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   technical_guide/index
   api/index
   releases
   contributing/index
   citing
