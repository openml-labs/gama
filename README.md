# GAMA
**G**enetic **A**utomated **M**achine learning **A**ssistant  
An automated machine learning tool based on genetic programming.  
Make sure to check out the [documentation](https://pgijsbers.github.io/gama/).

[![Build Status](https://travis-ci.org/PGijsbers/gama.svg?branch=master)](https://travis-ci.org/PGijsbers/gama)
[![codecov](https://codecov.io/gh/PGijsbers/gama/branch/master/graph/badge.svg)](https://codecov.io/gh/PGijsbers/gama)

-----------------------------------------------------------------------------------------------------------------------

GAMA is an AutoML package for end-users and AutoML researchers.
It uses genetic programming to efficiently generate optimized machine learning pipelines given specific input data and resource constraints.
A machine learning pipeline contains data preprocessing (e.g. PCA, Forward Feature Selection) as well as a machine learning algorithm (e.g. Logistic Regression, Random Forests), with fine-tuned hyperparameter settings (e.g. number of trees in a Random Forest).

GAMA can also combine multiple tuned machine learning pipelines together into an ensemble, which on average should help model performance.
At the moment, GAMA is restricted to classification and regression problems on tabular data.

In addition to its general use AutoML functionality, GAMA aims to serve AutoML researchers as well.
During the optimization process, GAMA keeps an extensive log of progress made.
Using this log, insight can be obtained on the behaviour of the population of pipelines.
For example, it can produce a graph that shows pipeline fitness over time:
![graph of fitness over time](https://raw.githubusercontent.com/PGijsbers/gama/images/fitnessgraph.png)

For more examples and information on the visualization, see [this notebook](https://github.com/PGijsbers/gama/blob/master/notebooks/GAMA%20Log%20Parser.ipynb).

## Installing GAMA
Clone GAMA:

`git clone https://github.com/PGijsbers/gama.git`

Move to the GAMA directory (`cd gama`) and install:
`python setup.py install`

All done!

## Minimal Example
The following example uses AutoML to find a machine learning pipeline that classifies breast cancer as malign or benign.
See the documentation for examples in 
[classification](https://pgijsbers.github.io/gama/user_guide/index.html#classification),
[regression](https://pgijsbers.github.io/gama/user_guide/index.html#regression),
using [ARFF as input](https://pgijsbers.github.io/gama/user_guide/index.html#using-arff-files).
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from gama import GamaClassifier

if __name__ == '__main__':
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

    automl = GamaClassifier(max_total_time=180, keep_analysis_log=False)
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit(X_train, y_train)

    label_predictions = automl.predict(X_test)
    probability_predictions = automl.predict_proba(X_test)

    print('accuracy:', accuracy_score(y_test, label_predictions))
    print('log loss:', log_loss(y_test, probability_predictions))
    # the `score` function outputs the score on the metric optimized towards (by default, `log_loss`)
    print('log_loss', automl.score(X_test, y_test))
```
*note*: By default, GamaClassifier optimizes towards `log_loss`.
