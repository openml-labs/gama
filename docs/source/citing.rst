Papers
======
This page contains bibtex entries for each paper, as well as up-to-date code listings from each paper.
Unless you want to reference a specific paper, when citing GAMA please cite the `JOSS article <http://joss.theoj.org/papers/10.21105/joss.01132>`_.

GAMA: a General Automated Machine learning Assistant
----------------------------------------------------
Features GAMA 20.2.1

Bibtex will be added after publication.

Up-to-date Listings
*******************
Listing 1:

.. code-block:: Python

    from gama import GamaClassifier
    from gama.search_methods import AsynchronousSuccessiveHalving
    from gama.postprocessing import EnsemblePostProcessing

    automl = GamaClassifier(
        search=AsynchronousSuccessiveHalving(),
        post_processing=EnsemblePostProcessing()
    )
    automl.fit(X, y)
    automl.predict(X_test)
    automl.fit(X_test, y_test)


GAMA: Genetic Automated Machine learning Assistant
--------------------------------------------------
Features GAMA 19.01.0

.. code-block:: latex

    @article{Gijsbers2019,
      doi = {10.21105/joss.01132},
      url = {https://doi.org/10.21105/joss.01132},
      year  = {2019},
      month = {jan},
      publisher = {The Open Journal},
      volume = {4},
      number = {33},
      pages = {1132},
      author = {Pieter Gijsbers and Joaquin Vanschoren},
      title = {{GAMA}: Genetic Automated Machine learning Assistant},
      journal = {Journal of Open Source Software}
    }

Listings
********
This paper features no listings.