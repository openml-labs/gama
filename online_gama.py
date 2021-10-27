import numpy as np
import pandas as pd
import arff
import sklearn.metrics

#Data prep
a = arff.load(open('data_streams/SEA_Abrubt_5.arff', 'r'),encode_nominal=True)
B = np.array_split(pd.DataFrame(a["data"]),500)
#B[0]
X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]
X_test = B[1].iloc[:,0:-1]
y_test = B[1].iloc[:,-1]

from gama import GamaClassifier
from gama.search_methods import RandomSearch, AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing
from river import compose

#Gama online - initial pipeline optimization

cls = GamaClassifier(max_total_time=60,
                       search = RandomSearch(),
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                       store = 'all',
                       online_scoring = 'balanced_accuracy')

cls.fit(X,y)

"""
steps = iter(cls.model.steps.values())

final = next(steps)
river_model = compose.Pipeline(final[0][1])

for i in range(0,len(X)):
    river_model = river_model.learn_one(X.iloc[i],y[i])
"""
print(cls.model)
y_pred = []
for i in range(0, len(X_test)):
    y_pred.append(cls.model.predict_one(X_test.iloc[i].to_dict()))

b_acc = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)  # equivalent to ROC_AUC in binary case
acc = sklearn.metrics.accuracy_score(y_test, y_pred)
print("Test batch 1 - Balanced accuracy %f - Accuracy %f\n" % (b_acc, acc))

