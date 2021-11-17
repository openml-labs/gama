#Application script for automated River


#imports

import numpy as np
import pandas as pd
import arff


from river import metrics
from river.drift import EDDM
from river import neighbors
from river import ensemble
from river import preprocessing
from river import linear_model
from river import tree
from river import evaluate
from river import datasets
import pprint

#User parameters

data_loc = 'data_streams/electricity-normalized.arff'     #needs to be arff
initial_batch = 5000                            #initial set of samples to train automl
sliding_window = 1000                           #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric = metrics.Accuracy()              #river metric to evaluate online learning
drift_detector = EDDM()

#Data

B = pd.DataFrame(arff.load(open(data_loc, 'r'),encode_nominal=True)["data"])

X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]

model_1 = neighbors.KNNClassifier()
model_2 = preprocessing.StandardScaler() | linear_model.Perceptron()
model_3 = preprocessing.AdaptiveStandardScaler() | tree.HoeffdingAdaptiveTreeClassifier()
model_4 = tree.HoeffdingAdaptiveTreeClassifier()


dataset = datasets.Phishing()

metric = metrics.Accuracy()
backup_ensemble = ensemble.VotingClassifier([model_1, model_2, model_3])
evaluate.progressive_val_score(dataset, backup_ensemble, metric)
print("ensemble: ", metric)
print(backup_ensemble._get_params())
breakpoint()

backup_ensemble.models.append(model_4)
evaluate.progressive_val_score(dataset, backup_ensemble, metric)
print("ensemble: ", metric)
print(backup_ensemble._get_params())
breakpoint()

# for i in range(initial_batch+1,len(X)):
#     #Test then train - by onelene
#     y_pred = backup_ensemble.predict_one(X.iloc[i].to_dict())
#     online_metric = online_metric.update(y[i], y_pred)
#     backup_ensemble = backup_ensemble.learn_one(X.iloc[i].to_dict(), int(y[i]))
#
#     #Print performance every x interval
#     if i%1000 == 0:
#         print(f'Test batch - {i} with {online_metric}')
