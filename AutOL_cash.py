#to be called from AutOL_streams and automl part there to be removed.

#imports

from gama import GamaClassifier
from gama.search_methods import AsyncEA
from gama.search_methods import RandomSearch
from gama.search_methods import AsynchronousSuccessiveHalving
from gama.postprocessing import BestFitOnlinePostProcessing


#Algorithm selection and hyperparameter tuning

import sys
time = sys.argv[1]
measure = sys.argv[2]
alg = sys.argv[3]
X = sys.argv[4]
y = sys.argv[5]


cls = GamaClassifier(max_total_time=time,
                       scoring=measure,
                       search = alg,
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing()
                     )

cls.fit(X,y)
print(f'Initial model is {cls.model} and hyperparameters are: {cls.model._get_params()}')


#it should return trained cls.