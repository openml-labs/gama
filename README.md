# GAMA
**G**enetic **A**utomated **M**achine learning **A**ssistant  
An automated machine learning tool based on genetic programming.  
Make sure to check out the [documentation](https://pgijsbers.github.io/gama/).

[![Build Status](https://travis-ci.org/PGijsbers/gama.svg?branch=master)](https://travis-ci.org/PGijsbers/gama)
[![codecov](https://codecov.io/gh/PGijsbers/gama/branch/master/graph/badge.svg)](https://codecov.io/gh/PGijsbers/gama)

## Installing GAMA
Clone GAMA:

`git clone https://github.com/PGijsbers/gama.git`

Move to the GAMA directory (`cd gama`) and install:
`python setup.py install`

All done!

## Minimal Example
The following example uses AutoML to find a machine learning pipeline to classify images of digits.
See the documentation for examples in 
[classification](https://pgijsbers.github.io/gama/user_guide/index.html#classification),
[regression](https://pgijsbers.github.io/gama/user_guide/index.html#regression),
using [ARFF as input](https://pgijsbers.github.io/gama/user_guide/index.html#using-arff-files).
```
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
