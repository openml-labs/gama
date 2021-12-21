#Application script for automated River


#imports

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
from river.drift import EDDM
from river import evaluate
from river import stream
from river import ensemble
from river import datasets

from skmultiflow import drift_detection
import matplotlib.pyplot as plt
import wandb


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

           ]
#Metrics
gama_metrics = ['accuracy',              #0
                'balanced_accuracy',     #1
                'f1',                    #2
                'roc_auc',               #3
                'rmse']                  #4
online_metrics = [metrics.Accuracy(),               #0
           metrics.BalancedAccuracy(),              #1
           metrics.F1(),                            #2
           metrics.ROCAUC(),                        #3
           metrics.RMSE()]                          #4

#Search algorithms
search_algs = [RandomSearch(),                      #0
               AsyncEA(),                           #1
               AsynchronousSuccessiveHalving()]     #2
#User parameters

print(sys.argv[0]) # prints python_script.py
print(f"Data stream is {datasets[int(sys.argv[1])]}.")                      # prints dataset no
print(f"Initial batch size is {int(sys.argv[2])}.")                         # prints initial batch size
print(f"Sliding window size is {int(sys.argv[3])}.")                        # prints sliding window size
print(f"Gama performance metric is {gama_metrics[int(sys.argv[4])]}.")      # prints gama performance metric
print(f"Online performance metric is {online_metrics[int(sys.argv[5])]}.")  # prints online performance metric
print(f"Time budget for GAMA is {int(sys.argv[6])}.")                       # prints time budget for GAMA
print(f"Search algorithm for GAMA is {search_algs[int(sys.argv[7])]}.")     # prints search algorithm for GAMA

data_loc = datasets[int(sys.argv[1])]               #needs to be arff
initial_batch = int(sys.argv[2])                    #initial set of samples to train automl
sliding_window = int(sys.argv[3])                   #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric  = online_metrics[int(sys.argv[5])]   #river metric to evaluate online learning
#drift_detector = EDDM()                            #river drift detector - issues
drift_detector = drift_detection.EDDM()             #multiflow drift detector
live_plot = False

#Plot initialization
if live_plot:
    wandb.init(
        project="Ensemble-demo",
        config={
            "dataset": datasets[int(sys.argv[1])],
            "batch_size": int(sys.argv[2]),
            "sliding_window": int(sys.argv[3]),
            "gama_performance_metric": int(sys.argv[4]),
            "online_performance_metric": int(sys.argv[5]),
            "time_budget_gama": int(sys.argv[6]),
            "search_algorithm": int(sys.argv[7])
        })
#Data

B = pd.DataFrame(arff.load(open(data_loc, 'r'),encode_nominal=True)["data"])

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

#Algorithm selection and hyperparameter tuning

Auto_pipeline = GamaClassifier(max_total_time=int(sys.argv[6]),
                       scoring= gama_metrics[int(sys.argv[4])] ,
                       search = search_algs[int(sys.argv[7])],
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                       store='nothing',
                     )

Auto_pipeline.fit(X.iloc[0:initial_batch],y[0:initial_batch])
print(f'Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}')


#Online learning

Backup_ensemble = ensemble.VotingClassifier([Auto_pipeline.model])

Online_model = Auto_pipeline.model
count_drift = 0
last_training_point = initial_batch
for i in range(initial_batch+1,len(B)):
    #Test then train - by one
    y_pred = Online_model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    Online_model = Online_model.learn_one(X.iloc[i].to_dict(), int(y[i]))

    #Print performance every x interval

    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')
        if live_plot:
            wandb.log({"current_point": i, "Prequential performance": online_metric.get()})
    # Check for drift

    drift_detector.add_element(int(y_pred != y[i]))
    if (drift_detector.detected_change()) or ((i - last_training_point) > 50000):
        if i - last_training_point < 1000:
            continue
        if drift_detector.detected_change():
            print(f"Change detected at data point {i} and current performance is at {online_metric}")
            if live_plot:
                wandb.log({"drift_point": i, "current_point": i, "Prequential performance": online_metric.get()})
        if (i - last_training_point) > 50000:
            print(f"No drift but retraining point {i} and current performance is at {online_metric}")
            if live_plot:
                wandb.log({"current_point": i, "Prequential performance": online_metric.get()})

        last_training_point = i

        #Sliding window at the time of drift
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)

        #re-optimize pipelines with sliding window
        Auto_pipeline = GamaClassifier(max_total_time=int(sys.argv[6]),
                                       scoring= gama_metrics[int(sys.argv[4])],
                                       search=search_algs[int(sys.argv[7])],
                                       online_learning=True,
                                       post_processing=BestFitOnlinePostProcessing(),
                                       store='nothing',
                                       )
        Auto_pipeline.fit(X_sliding, y_sliding)

        #Ensemble performance comparison
        dataset = []
        for xi, yi in stream.iter_pandas(X_sliding, y_sliding):
            dataset.append((xi, yi))

        Perf_ensemble = evaluate.progressive_val_score(dataset, Backup_ensemble, metrics.Accuracy())
        Perf_automodel = evaluate.progressive_val_score(dataset, Auto_pipeline.model, metrics.Accuracy())

        if Perf_ensemble.get() > Perf_automodel.get():
            Online_model = Backup_ensemble
            print("Online model is updated with Backup Ensemble.")
        else:
            Online_model = Auto_pipeline.model
            print("Online model is updated with latest AutoML pipeline.")

        #Ensemble update with new model, remove oldest model if ensemble is full
        Backup_ensemble.models.append(Auto_pipeline.model)
        if len(Backup_ensemble.models) > 10:
            Backup_ensemble.models.pop(0)

        print(f'Current model is {Online_model} and hyperparameters are: {Online_model._get_params()}')


