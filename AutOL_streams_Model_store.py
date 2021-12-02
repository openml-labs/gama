#Application script for automated River


#imports

import subprocess
import numpy as np
import pandas as pd
import arff

from gama import GamaClassifier
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing
from multiprocessing import Pool
from river import metrics
from river.drift import EDDM
from river import evaluate
from river import stream
from river import ensemble
#User parameters

data_loc = 'data_streams/electricity-normalized.arff'     #needs to be arff
initial_batch = 5000                            #initial set of samples to train automl
sliding_window = 1000                           #update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
online_metric = metrics.Accuracy()              #river metric to evaluate online learning
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

cls = GamaClassifier(max_total_time=10,
                       scoring='accuracy',
                       search = AsyncEA(),
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                       #store = 'all'
                     )

cls.fit(X.iloc[0:initial_batch],y[0:initial_batch])
print(f'Initial model is {cls.model} and hyperparameters are: {cls.model._get_params()}')

#Online learning
model_store = []
for i in range(initial_batch+1,len(X)):
    #Test then train - by one
    y_pred = cls.model.predict_one(X.iloc[i].to_dict())
    online_metric = online_metric.update(y[i], y_pred)
    cls.model = cls.model.learn_one(X.iloc[i].to_dict(), int(y[i]))
    #Print performance every x interval
    if i%1000 == 0:
        print(f'Test batch - {i} with {online_metric}')

    #Check for drift
    in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
    if in_drift:
        # Functions can also be used but not doing them not to maintain homogenity in experimental code
        # model_store,max_model = model_store_computation(model_store, i, X, y)
        # cls = classifier_search_gama(X, y)
        #
        # print(cls)
        #Sliding window at the time of drift
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)
        if len(model_store) > 2:
            score_arr = []

            for i in range(len(model_store)):
                score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), model_store[i],
                                               metrics.Accuracy())
                score_arr.append(score.get())
            print(score_arr)

        curr_model_score = evaluate.progressive_val_score(stream.iter_pandas(X_sliding, y_sliding), cls.model, metrics.Accuracy())
        print(curr_model_score.get())
        if len(model_store) < 5:
            model_store.append(cls.model)
        elif curr_model_score.get() > any(score_arr):
            low_model_score = min(score_arr)
            low_model = score_arr.index(low_model_score)
            model_store = model_store.pop(low_model)
            model_store.append(cls.model)
        print(f"Change detected at data point {i} and current performance is at {online_metric}")
        #re-optimize pipelines with sliding window
        cls = GamaClassifier(max_total_time=180,
                         scoring='accuracy',
                         search=AsyncEA(),
                         online_learning=True,
                         post_processing=BestFitOnlinePostProcessing(),
                         # store='all'
                         )

        X_sliding = X.iloc[(i - sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i - sliding_window):i].reset_index(drop=True)

        cls.fit(X_sliding, y_sliding)
        print(f'Current model is {cls.model} and hyperparameters are: {cls.model._get_params()}')

