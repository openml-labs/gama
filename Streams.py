#Application script for automated River


#imports

import numpy as np
import pandas as pd
import arff

from skmultiflow import drift_detection

from river import metrics
from river.drift import EDDM
from river import neighbors
from river import ensemble
from river import preprocessing
from river import linear_model
from river import tree
from river import evaluate
from river import datasets
from river import stream

import matplotlib.pyplot as plt
plt.ion() ## Note this correction
fig=plt.figure()
#plt.axis([0,10000,0,1])
x_plot=[]
y_plot=[]
plt.show(block=False)

#User parameters

data_loc = 'data_streams/HYPERPLANE_01.arff'    #needs to be arff
initial_batch = 5000                            #initial set of samples to train automl
sliding_window = 1000                           #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric = metrics.Accuracy()              #river metric to evaluate online learning
#drift_detector = EDDM()
drift_detector = drift_detection.EDDM()

#Data

B = pd.DataFrame(arff.load(open(data_loc, 'r'),encode_nominal=True)["data"])
B = B[~((B.iloc[:,0:-1] == 0).any(axis=1))].reset_index(drop=True)

X = B.iloc[:,0:-1]
y = B.iloc[:,-1]

model_1 = tree.ExtremelyFastDecisionTreeClassifier()
model_2 = preprocessing.StandardScaler() | linear_model.Perceptron()
model_3 = preprocessing.AdaptiveStandardScaler() | tree.HoeffdingAdaptiveTreeClassifier()
model_4 = tree.HoeffdingAdaptiveTreeClassifier()
model_5 = ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingAdaptiveTreeClassifier())

model = model_5

for i in range(initial_batch+1,len(X)):
    #Test then train - by one
    y_pred = model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    model = model.learn_one(X.iloc[i].to_dict(), int(y[i]))

    #Print performance every x interval
    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')
        x_plot.append(i)
        y_plot.append(online_metric.get())
        plt.plot(x_plot, y_plot)
        plt.draw()
        plt.pause(0.0001)

    #Check for drift
    #in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
    drift_detector.add_element(int(y_pred!=y[i]))
    #if in_drift:
    if drift_detector.detected_change():
        print(f"Change detected at data point {i} and current performance is at {online_metric}")
        #drift_detector.reset()


# #dataset = datasets.Phishing()
# dataset = []
# for xi, yi in stream.iter_pandas(X, y):
#     dataset.append((xi,yi))
#
# metric = metrics.Accuracy()
# backup_ensemble = ensemble.VotingClassifier([model_1, model_2, model_3])
# evaluate.progressive_val_score(dataset, backup_ensemble, metric)
# print("ensemble: ", metric)
# print(backup_ensemble._get_params())



# for i in range(initial_batch+1,len(X)):
#     #Test then train - by onelene
#     y_pred = backup_ensemble.predict_one(X.iloc[i].to_dict())
#     online_metric = online_metric.update(y[i], y_pred)
#     backup_ensemble = backup_ensemble.learn_one(X.iloc[i].to_dict(), int(y[i]))
#
#     #Print performance every x interval
#     if i%1000 == 0:
#         print(f'Test batch - {i} with {online_metric}')
