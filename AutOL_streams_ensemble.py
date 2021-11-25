#Application script for automated River


#imports

import numpy as np
import pandas as pd
import arff

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

#User parameters

import sys
print(sys.argv[0]) # prints python_script.py
print(sys.argv[1]) # prints dataset no
print(sys.argv[2]) # prints initial batch size
print(sys.argv[3]) # prints sliding window size
print(sys.argv[4]) # prints online metric

data_loc = 'data_streams/electricity-normalized.arff'     #needs to be arff
initial_batch = int(sys.argv[1])                            #initial set of samples to train automl
sliding_window = 3000                           #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric = metrics.Accuracy()              #river metric to evaluate online learning
drift_detector = EDDM()

#Data

B = pd.DataFrame(arff.load(open(data_loc, 'r'),encode_nominal=True)["data"])

X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]

#Algorithm selection and hyperparameter tuning

Auto_pipeline = GamaClassifier(max_total_time=60,
                       scoring='accuracy',
                       search = RandomSearch(),
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                     )

Auto_pipeline.fit(X.iloc[0:initial_batch],y[0:initial_batch])
print(f'Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}')


#Online learning

Backup_ensemble = ensemble.VotingClassifier([Auto_pipeline.model])

Online_model = Auto_pipeline.model
for i in range(initial_batch+1,len(X)):
    #Test then train - by one
    y_pred = Online_model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    Online_model = Online_model.learn_one(X.iloc[i].to_dict(), int(y[i]))

    #Print performance every x interval
    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')

    #Check for drift
    in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
    if in_drift:
        print(f"Change detected at data point {i} and current performance is at {online_metric}")

        #Sliding window at the time of drift
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)

        #re-optimize pipelines with sliding window
        Auto_pipeline = GamaClassifier(max_total_time=60,
                             scoring='accuracy',
                             search=RandomSearch(),
                             online_learning=True,
                             post_processing=BestFitOnlinePostProcessing(),
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



