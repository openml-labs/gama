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

#Algorithm selection and hyperparameter tuning

cls = GamaClassifier(max_total_time=60,
                       scoring='accuracy',
                       search = AsyncEA(),
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                       #store = 'all'
                     )

cls.fit(X.iloc[0:initial_batch],y[0:initial_batch])
print(f'Initial model is {cls.model} and hyperparameters are: {cls.model._get_params()}')

#Online learning

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
        print(f"Change detected at data point {i} and current performance is at {online_metric}")
        #re-optimize pipelines with sliding window
        cls = GamaClassifier(max_total_time=60,
                             scoring='accuracy',
                             search=AsyncEA(),
                             online_learning=True,
                             post_processing=BestFitOnlinePostProcessing(),
                             store='all')
        X_sliding = X.iloc[(i-sliding_window):i].reset_index(drop=True)
        y_sliding = y[(i-sliding_window):i].reset_index(drop=True)
        cls.fit(X_sliding, y_sliding)
        print(f'Current model is {cls.model} and hyperparameters are: {cls.model._get_params()}')



