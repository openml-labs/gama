# GAMA
**G**enetic **A**utomated **M**achine learning **A**ssistant  
An automated machine learning tool based on genetic programming.

[![Build Status](https://travis-ci.org/PGijsbers/gama.svg?branch=master)](https://travis-ci.org/PGijsbers/gama)

**WORK IN PROGRESS**

Notice from author: 
I am still working on getting things in order for a 'proper' public release.
Things left to do:
 - Set up CI
 - Review and expand documentation
 - last round of refactoring, time permitting

## Installing GAMA
Clone GAMA:

`git clone https://github.com/PGijsbers/gama.git`

Move to the GAMA directory (`cd gama`) and install:
`python setup.py install`

All done!

## Minimal Example
The following example uses AutoML to find a machine learning pipeline to classify images of digits.
```
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gama import GamaClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, random_state=42)

automl = GamaClassifier(max_total_time=300, n_jobs=4)
automl.fit(X_train, y_train)
predictions = automl.predict(X_test)
print('accuracy', accuracy_score(y_test, predictions))
```

*note*: By default, GamaClassifier optimizes towards `log_loss`.
