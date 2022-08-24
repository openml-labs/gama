# Application script for automated River


# imports

import numpy as np
import pandas as pd
import arff

from skmultiflow import drift_detection

from river.drift import EDDM
from river import neighbors
from river import ensemble
from river import preprocessing
from river import linear_model
from river import tree
from river import evaluate
from river import datasets
from river import stream
from river import naive_bayes
from sklearn.ensemble import GradientBoostingClassifier

import wandb

from sklearn import metrics

# User parameters

data_loc = "data_streams/SEA_Abrubt_5.arff"  # needs to be arff
initial_batch = 1000  # initial set of samples to train automl
# drift_detector = EDDM()
drift_detector = drift_detection.EDDM()
live_plot = True

# Plot initialization
if live_plot:
    wandb.init(
        project="Ensemble-demo",
        entity="autoriver",
        config={
            "dataset": data_loc,
        },
    )


# Data

B = pd.DataFrame(arff.load(open(data_loc, "r"), encode_nominal=True)["data"])
B = B[~((B.iloc[:, 0:-1] == 0).any(axis=1))].reset_index(drop=True)

X = B.iloc[:, 0:-1]
y = B.iloc[:, -1]

# model = preprocessing.StandardScaler() | linear_model.Perceptron()
model = naive_bayes.BernoulliNB()

# initial training
# breakpoint()
model = model.learn_many(X.iloc[0:initial_batch], y[0:initial_batch])

for i in range(initial_batch + 1, len(X), 1000):
    # Test then train - by one
    y_pred = 1 * model.predict_many(X.iloc[i : i + 1000])
    y_pred = y_pred.astype(int)
    performance = metrics.accuracy_score(y[i : i + 1000], y_pred)
    print(f"Test batch - {i} with {performance}")
    model = model.learn_many(X.iloc[i : i + 1000], y[i : i + 1000])

    # Print performance every x interval
    for j in range(i, i + 999):
        if live_plot:
            wandb.log({"current_point": j, "Batch performance": performance})
        drift_detector.add_element(int(y_pred[j] != int(y[j])))
        if drift_detector.detected_change():
            print(
                f"Change detected at data point {j} and current performance is at {performance}"
            )
            if live_plot:
                wandb.log(
                    {
                        "drift_point": j,
                        "current_point": j,
                        "Prequential performance": performance,
                    }
                )

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
