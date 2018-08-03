# GAMA Basics
This file gives an introduction to basic components and concepts of GAMA.
GAMA is an AutoML tool which aims to automatically find the right machine learning algorithms to create the best possible data for your model.

In the process, GAMA performs a search over *machine learning pipelines*.
An example of a machine learnig pipeline would be to first perform data normalization and then use a decision tree classifier to make a prediction on the normalized data.
More formally, a *machine learning pipeline* is a sequence of one or more *components*.
A *component* is an algorithm which performs either data transformation *or* a prediction.
This means that components can be preprocessing algorithms such as PCA or standard scaling, or a predictor such as a decision tree or support vector machine.
A machine learning pipeline then consists of zero or more preprocessing components followed by a predictor component.

Given some data, GAMA will start a search to try and find the best possible machine learning pipelines for it.
After the search, the best model found can be used to make predictions.
Alternatively, GAMA can combine several models into an *ensemble* to take into account more than one model when making predictions.


## Minimal Example
GAMA is compliant with the scikit-learn interface (**except there is no score yet**).
An example which runs GAMA on a classification problem can be seen below.

```python
from gama import GamaClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.targets, shuffle=True)

clf = GamaClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
```

In a similar fashion regression data can be handled by importing and calling `GamaRegressor` instead.

Some optional hyperparameters allow you to tweak the way certain parts of optimization are done.
Below, you will find an example with a few more hyperparameters exposed.
For full details on available hyperparameters, read more detailed documentation [here](). (**todo: write documentation**)


```python
from gama import GamaClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.targets, shuffle=True)

clf = GamaClassifier(objectives=('f1_macro', 'size'),
                    population_size=50,
                    n_jobs=8,
                    random_state=0,
                    max_total_time=3600,
                    max_eval_time=300)
clf.fit(X_train, y_train, auto_ensemble_n=25, keep_cache=True)
y_predict = clf.predict(X_test)
```

In this example, the following hyperparameters are changed from their default values:
 - `objectives`: states towards which metrics to optimize. Supplying a binary tuple allows for multi-objective optimization. Here the search aims to maximize `f1_macro` and minimize "size" (i.e. the number of components in the pipeline).
 - `population_size`: a hyperparameter for the evolutionary algorithm underlying GAMA which indicates the number of individuals to keep in the population at any one time.
 - `n_jobs`: determines how many processes can be run in parallel during `fit`.
 - `random_state`: seeds all random decisions, such that a run can be fully reproducible. However, reproducibility is only guaranteed for `n_jobs=1`.
 - `max_total_time`: the maximum time in seconds that GAMA should aim to use to construct a model from the data.
 - `max_eval_time`: the maximum time in seconds that GAMA is allowed to use to evaluate a single machine learning pipeline.
 
 And the `fit` call also contains two new hyperparameters:
  -  `auto_ensemble_n`: This hyperparameter specifies the number of models to include in the final ensemble.
 The final ensemble may contain duplicates as a form of assigning weights.
  - `keep_cache`: During the optimization process, each model evaluated is stored alongside its predictions. 
   This is useful for automatically constructing an ensemble.
   Normally, this cache is deleted automatically, but should you wish to keep it, you can specify it here.
   
## Setup
Currently, GAMA can not be installed like a regular python package.
To use GAMA, either add the directory in which it resides to the PATH
```python
import sys
sys.path.append('D:\\example\\path\\gama')
```

or place it in the same folder as the script you wish to call it from.

- [MyDirectory]
  * my_script.py
  - [gama]

   
 
 