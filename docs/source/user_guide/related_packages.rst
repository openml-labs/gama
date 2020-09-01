:orphan:

Related Packages
----------------

There are many different AutoML packages in Python and other languages.
`This webpage <https://openml.github.io/automlbenchmark/automl_overview.html>`_ of the AutoML Benchmark project has an overview of commonly used open source AutoML frameworks.
Another nice overview of papers and projects is gathered at `this Github repo <https://github.com/hibayesian/awesome-automl-papers#projects>`_.

.. note::
    Any comparisons in this section are intended to highlight the main differences, but are by no means exhaustive.
    If anything is missing that is important to add, please open an issue.

TPOT
****
`TPOT <https://epistasislab.github.io/tpot/>`_ is the package most closely related to GAMA.
Like GAMA, it is an AutoML package that is based on genetic programming in Python.
Below you will find a list of differences and similarities of the packages.

The evolution algorithm differs.
TPOT uses a synchronous form of evolution, whereas GAMA uses asynchronous evolution (by default).
The advantage of asynchronous evolution is that in theory it is able to utilize computing resources better.
This is due to generation-based evolution waiting for the last individual in the population to be evaluated in each generation,
whereas asynchronous evolution does not have any point where all evaluations need to be finished at the same time.

TPOT sports some great user-friendly features such as a less limited command-line interface
and `DASK <https://dask.org/>`_ integration (which allows the use of a Dask cluster for pipeline evaluations and further optimizations).

GAMA focuses more on extensibility and research friendliness,
with the ability to configure your AutoML pipeline (for instance using a different search strategy),
an ensembling post-processing step, visualization of the optimization process and support for `ARFF <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_ files.


auto-sklearn
************
`auto-sklearn <https://automl.github.io/auto-sklearn/stable/>`_ is another AutoML package in Python.
It uses Bayesian optimization to search for good machine learning pipelines and also features automatic ensemble construction.
Auto-sklearn won ChaLearn AutoML challenge 1 and 2.
