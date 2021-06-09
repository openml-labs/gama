# GAMA

**G**eneral **A**utomated **M**achine learning **A**ssistant  
An automated machine learning tool based on genetic programming.  
Make sure to check out the [documentation](https://openml-labs.github.io/gama/).

[![Build Status](https://travis-ci.org/openml-labs/gama.svg?branch=master)](https://travis-ci.org/openml-labs/gama)
[![codecov](https://codecov.io/gh/openml-labs/gama/branch/master/graph/badge.svg)](https://codecov.io/gh/openml-labs/gama)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.01132/status.svg)](https://doi.org/10.21105/joss.01132)

---

GAMA is an AutoML package for end-users and AutoML researchers.
It generates optimized machine learning pipelines given specific input data and resource constraints.
A machine learning pipeline contains data preprocessing (e.g. PCA, normalization) as well as a machine learning algorithm (e.g. Logistic Regression, Random Forests), with fine-tuned hyperparameter settings (e.g. number of trees in a Random Forest).

To find these pipelines, multiple search procedures have been implemented.
GAMA can also combine multiple tuned machine learning pipelines together into an ensemble, which on average should help model performance.
At the moment, GAMA is restricted to classification and regression problems on tabular data.

In addition to its general use AutoML functionality, GAMA aims to serve AutoML researchers as well.
During the optimization process, GAMA keeps an extensive log of progress made.
Using this log, insight can be obtained on the behaviour of the search procedure.
For example, it can produce a graph that shows pipeline fitness over time:
![graph of fitness over time](https://github.com/openml-lab/gama/blob/master/docs/source/technical_guide/images/viz.gif)

For more examples and information on the visualization, see [the technical guide](https://openml-labs.github.io/gama/master/user_guide/index.html#dashboard).

## Installing GAMA

You can install GAMA with pip: `pip install gama`

## Minimal Example

The following example uses AutoML to find a machine learning pipeline that classifies breast cancer as malign or benign.
See the documentation for examples in
[classification](https://openml-labs.github.io/gama/master/user_guide/index.html#classification),
[regression](https://openml-labs.github.io/gama/master/user_guide/index.html#regression),
using [ARFF as input](https://openml-labs.github.io/gama/master/user_guide/index.html#using-arff-files).

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    automl = GamaClassifier(max_total_time=180, store="nothing")
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print('accuracy:', accuracy_score(y_test, label_predictions))
    print('log loss:', log_loss(y_test, probability_predictions))
    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
    print('log_loss', automl.score(X_test, y_test))
```

_note_: By default, GamaClassifier optimizes towards `log_loss`.

## Citing

If you want to cite GAMA, please use [our JOSS publication](http://joss.theoj.org/papers/10.21105/joss.01132).

```latex
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
```
