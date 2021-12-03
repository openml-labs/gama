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

#Datasets
datasets =['data_streams/electricity-normalized.arff',      #0
           'data_streams/new_airlines.arff',                #1
           'data_streams/new_IMDB_drama.arff',              #2
           'data_streams/SEA_Abrubt_5.arff',                #3
           'data_streams/HYPERPLANE_01.arff',               #4
           'data_streams/SEA_Mixed_5.arff',                 #5
           'data_streams/Forestcover.arff',                 #6
           'data_streams/new_ldpa.arff',                    #7
           'data_streams/new_pokerhand-normalized.arff',    #8
           'data_streams/new_Run_or_walk_information.arff', #9
           ]
#Metrics
gama_metrics = ['accuracy',              #0
                'balanced_accuracy',     #1
                'f1',                    #2
                'roc_auc',               #3
                'rmse']


online_metrics = [metrics.Accuracy(),               #0
           metrics.BalancedAccuracy(),              #1
           metrics.F1(),                            #2
           metrics.ROCAUC(),                        #3
           metrics.RMSE()]                          #4

#Search algorithms
search_algs = [RandomSearch(),                      #0
               AsyncEA(),                           #1
               AsynchronousSuccessiveHalving()]

#User parameters
print(sys.argv[0]) # prints python_script.py
print(f"Data stream is {datasets[int(sys.argv[1])]}.")                      # prints dataset no
print(f"Initial batch size is {int(sys.argv[2])}.")                         # prints initial batch size
print(f"Sliding window size is {int(sys.argv[3])}.")                        # prints sliding window size
print(f"Gama performance metric is {gama_metrics[int(sys.argv[4])]}.")      # prints gama performance metric
print(f"Online performance metric is {online_metrics[int(sys.argv[5])]}.")  # prints online performance metric
print(f"Time budget for GAMA is {int(sys.argv[6])}.")                       # prints time budget for GAMA
print(f"Search algorithm for GAMA is {search_algs[int(sys.argv[7])]}.")


data_loc = datasets[int(sys.argv[1])]               #needs to be arff
initial_batch = int(sys.argv[2])                    #initial set of samples to train automl
sliding_window = int(sys.argv[3])                   #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric  = online_metrics[int(sys.argv[5])]   #river metric to evaluate online learning
drift_detector = EDDM()

#Data
# def classifier_search_gama(X,y):
#     cls = GamaClassifier(max_total_time=180,
#                          scoring='accuracy',
#                          search=AsyncEA(),
#                          online_learning=True,
#                          post_processing=BestFitOnlinePostProcessing(),
#                          # store='all'
#                          )
#
#     X_sliding = X.iloc[(i - sliding_window):i].reset_index(drop=True)
#     y_sliding = y[(i - sliding_window):i].reset_index(drop=True)
#
#     cls.fit(X_sliding, y_sliding)
#     print(f'Current model is {cls.model} and hyperparameters are: {cls.model._get_params()}')
#     return cls
#
# def model_store_computation(model_store, i, X, y):
#     print(i)
#     X_sliding = X.iloc[(i - sliding_window):i].reset_index(drop=True)
#     y_sliding = y[(i - sliding_window):i].reset_index(drop=True)
#     if len(model_store) > 2:
#         score_arr = []
#
#         for j in range(len(model_store)):
#             score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), model_store[j],
#                                                    metrics.Accuracy())
#             score_arr.append(score.get())
#         print(score_arr)
#
#     curr_model_score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), cls.model,
#                                                       metrics.Accuracy())
#     print(curr_model_score.get())
#     if len(model_store) < 5:
#         model_store.append(cls.model)
#     elif curr_model_score.get() > any(score_arr):
#         print('Current model added to Model Store')
#         low_model_score = min(score_arr)
#         low_model = score_arr.index(low_model_score)
#         model_store = model_store.pop(low_model)
#         model_store.append(cls.model)
#     max_score = max(score_arr)
#     max_model_index = score_arr.index(max_score)
#     max_model = model_store[max_model_index]
#     return model_store, max_score, max_model, curr_model_score



B = pd.DataFrame(arff.load(open(data_loc, 'r'),encode_nominal=True)["data"])

X = B[:].iloc[:,0:-1]
y = B[:].iloc[:,-1]

#Algorithm selection and hyperparameter tuning

Auto_pipeline = GamaClassifier(max_total_time=int(sys.argv[6]),
                       scoring= gama_metrics[int(sys.argv[4])] ,
                       search = search_algs[int(sys.argv[7])],
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                     )

Auto_pipeline.fit(X.iloc[0:initial_batch],y[0:initial_batch])
print(f'Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}')

cls = Auto_pipeline.model
#Online learning
model_store = []
for i in range(initial_batch+1,len(X)):
    #Test then train - by one
    y_pred = cls.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    cls = cls.learn_one(X.iloc[i].to_dict(), int(y[i]))
    #Print performance every x interval
    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')

    #Check for drift
    in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
    if in_drift:
        print(f"Change detected at data point {i} and current performance is at {online_metric}")
        # Functions can also be used but not doing them not to maintain homogenity in experimental code
        # model_store,max_model = model_store_computation(model_store, i, X, y)
        # cls = classifier_search_gama(X, y)
        #
        # print(cls)
        #Sliding window at the time of drift
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)

        Auto_pipeline = GamaClassifier(max_total_time=int(sys.argv[6]),
                                       scoring= gama_metrics[int(sys.argv[4])],
                                       search=search_algs[int(sys.argv[7])],
                                       online_learning=True,
                                       post_processing=BestFitOnlinePostProcessing(),
                                       )
        Auto_pipeline.fit(X_sliding, y_sliding)

        if len(model_store) > 5:
            print('')
            score_arr = []

            for i in range(len(model_store)):
                score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), model_store[i],
                                                       metrics.Accuracy())
                score_arr.append(score.get())

        curr_model_score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), cls.model, metrics.Accuracy())
        print(curr_model_score.get())

        if len(model_store) < 5:
            print('current model added to model store')
            model_store.append(cls.model)
        elif curr_model_score.get() > any(score_arr):
            low_model_score = min(score_arr)
            low_model = score_arr.index(low_model_score)
            model_store = model_store.pop(low_model)
            model_store.append(cls.model)
        max_score = max(score_arr)
        max_model_index = score_arr.index(max_score)
        max_model = model_store[max_model_index]

        automl_score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), Auto_pipeline.model, metrics.Accuracy())
        if automl_score_score > max_score:
            print("Online model is updated with latest AutoML pipeline.")
            cls = Auto_pipeline.model
        elif automl_score < max_score:
            print("Online model is updated with Model Store pipeline.")
            cls = max_model

        print(f"Change detected at data point {i} and current performance is at {online_metric}")
        #re-optimize pipelines with sliding window

        print(f'Current model is {cls.model} and hyperparameters are: {cls.model._get_params()}')

