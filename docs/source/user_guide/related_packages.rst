Related Packages
----------------

TPOT
****
`TPOT <https://epistasislab.github.io/tpot/>`_ is the package most closely related to GAMA.
Like GAMA, it is an AutoML package that is based on genetic programming in Python.
Below you will find a list of differences and similarities of the packages.
*Disclaimer: the list is intended to give the main differences, but is by no means exhaustive. If anything is missing that
 is important to add, please open an issue.*

In GAMA but not in TPOT:

 * Asynchronous evolution. In theory, asynchronous evolution should be able to utilize the computing resources better.
 This is due to generation-based evolution waiting for the last individual in the population to be evaluated in each generation,
 whereas asynchronous evolution does not have any point where all evaluations need to be finished at the same time.
 More on the differences in the user guide.

 * ARFF support. The Attribute-Relation File Format (`ARFF <https://www.cs.waikato.ac.nz/ml/weka/arff.html>`_)
 is used to store and describe data. It can be used, for instance, to make sure certain data preprocessing steps,
 such as category encoding, are applied only on the columns it was intended for instead of using heuristics.

 * Automatic Ensembling. During the optimization process, many pipelines get evaluated. Instead of using only
 the single best in the final model to make predictions, GAMA can create an ensemble out of a subset of them.
 This can lead to better generalization performance.

In TPOT but not in GAMA:

 * Code export. TPOT is able to export Python code to recreate the best found pipeline. The Python code can be used
 without any further dependency on TPOT.

 * Command Line Interface. TPOT can be used directly from the command line (in addition to the Python API).

 * DASK integration. When DASK is installed, TPOT's pipelines can be evaluated on any DASK cluster. Moreover, it can
 use other optimizations such as caching and computing graphs to avoid (some of the) double work.

An empirical study of performance will follow.

auto-sklearn
************
`auto-sklearn <https://automl.github.io/auto-sklearn/stable/>`_ is another AutoML package in Python.
It uses Bayesian optimization to search for good machine learning pipelines and also features automatic ensemble construction.
Auto-sklearn won ChaLearn AutoML challenge 1 and 2.

Other packages
**************
[I will provide a link here]