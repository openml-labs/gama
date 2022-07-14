import numpy as np
import pandas as pd
import arff
import sys

from gama import GamaClassifier
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing

from river import metrics
from river import evaluate
from river import stream
from river import ensemble


file = open('data_streams/electricity-normalized.arff')
B = pd.DataFrame(arff.loads(file, encode_nominal=True)["data"])
X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]
X_train = X.iloc[0:1000]
y_train = y[0:1000]
X_test = X.iloc[1001:2000]
y_test = y[1001:2000]

automl =  GamaClassifier()
automl.fit(X_train, y_train)
automl.predict(X_test)
automl.predict_proba(X_test)

print("accuracy:", accuracy_score(y_test, label_predictions))
print("log loss:", log_loss(y_test, probability_predictions))