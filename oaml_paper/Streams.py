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
from river import naive_bayes
from river import evaluate
from river import datasets
from river import stream
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import wandb
import sys

#Datasets
datasets =['data_streams/electricity-normalized.arff',      #0
           'data_streams/new_airlines.arff',                #1
           'data_streams/new_IMDB_drama.arff',              #2      - target at the beginning
           'data_streams/new_vehicle_sensIT.arff',          #3      - target at the beginning
           'data_streams/SEA_Abrubt_5.arff',                #4
           'data_streams/HYPERPLANE_01.arff',               #5
           'data_streams/SEA_Mixed_5.arff',                 #6
           'data_streams/Forestcover.arff',                 #7      - for later
           'data_streams/new_ldpa.arff',                    #8      - for later
           'data_streams/new_pokerhand-normalized.arff',    #9      - for later
           'data_streams/new_Run_or_walk_information.arff', #10     - for later
           'data_streams/New_dirty_data/Activity_raw.arff', #11
           'data_streams/New_dirty_data/Connect_4.arff',    #12
           ]
#Models

model_1 = tree.ExtremelyFastDecisionTreeClassifier()
model_2 = preprocessing.StandardScaler() | linear_model.Perceptron()
model_3 = preprocessing.AdaptiveStandardScaler() | tree.HoeffdingAdaptiveTreeClassifier()
model_4 = tree.HoeffdingAdaptiveTreeClassifier()
model_5 = ensemble.LeveragingBaggingClassifier(preprocessing.StandardScaler() | linear_model.Perceptron())
model_6 = preprocessing.StandardScaler() | neighbors.KNNClassifier()
model_7 = naive_bayes.BernoulliNB()

model = model_3

#User parameters

#User parameters

print(sys.argv[0]) # prints python_script.py
print(f"Data stream is {datasets[int(sys.argv[1])]}.")                      # prints dataset no
print(f"Initial batch size is {int(sys.argv[2])}.")                         # prints initial batch size

data_loc = datasets[int(sys.argv[1])]                       #needs to be arff
initial_batch = int(sys.argv[2])                            #initial set of samples to train automl
online_metric = metrics.Accuracy()                          #river metric to evaluate online learning
#drift_detector = EDDM()
drift_detector = drift_detection.EDDM()
live_plot = False

#Plot initialization
if live_plot:
    wandb.init(
        project="Baseline-1 LeverageBagging-2",
        entity = "autoriver",
        config={
            "dataset": data_loc,
            "online_performance_metric": online_metric,
        })


#Data

file = open(data_loc)
B = pd.DataFrame(arff.loads(file, encode_nominal=True)["data"])
breakpoint()

# Preprocessing of data: Drop NaNs, move target to the end, check for zero values

if int(sys.argv[1]) in [2,3]:
    columns = B.columns.values.tolist()
    columns.remove(0)
    columns.append(0)
    B = B.reindex(columns, axis=1)

if pd.isnull(B.iloc[:, :]).any().any():
    print("Data X contains NaN values. The rows that contain NaN values will be dropped.")
    B.dropna(inplace=True)

if B[:].iloc[:,0:-1].eq(0).any().any():
    print("Data contains zero values. They are not removed but might cause issues with some River learners.")

X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]

#initial training
for i in range(0,initial_batch):
    model = model.learn_one(X.iloc[i].to_dict(), int(y[i]))


for i in range(initial_batch+1,len(X)):
    #Test then train - by one
    y_pred = model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    model = model.learn_one(X.iloc[i].to_dict(), int(y[i]))

    #Print performance every x interval
    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')
        if live_plot:
            wandb.log({"current_point": i, "Prequential performance": online_metric.get()})

    # #Check for drift
    # #in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
    # drift_detector.add_element(int(y_pred != y[i]))
    # #if in_drift:
    # if drift_detector.detected_change():
    #     print(f"Change detected at data point {i} and current performance is at {online_metric}")
    #     if live_plot:
    #         wandb.log({"drift_point": i, "current_point": i, "Prequential performance": online_metric.get()})

