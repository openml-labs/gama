import numpy as np
import pandas as pd
import arff
import sklearn.metrics

#Data prep
a = arff.load(open('/home/bcelik/SEA_Abrubt_5.arff', 'r'),encode_nominal=True)
B = np.array_split(pd.DataFrame(a["data"]),500)
#B[0]
X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]

from gama import GamaClassifier
from gama.search_methods import RandomSearch
from gama.postprocessing import BestFitOnlinePostProcessing

#Gama online - initial pipeline optimization

cls = GamaClassifier(max_total_time=60,
                       #scoring='accuracy',
                       search = RandomSearch(),
                       online_learning = True,
                       post_processing = BestFitOnlinePostProcessing(),
                       store = 'all')

cls.fit(X,y)

#print(cls.model)

for i in range(0,len(X)):
  cls.model.learn_one(X.iloc[i],y[i], learn_unsupervised = False)

#y_pred = model.predict(B[1].iloc[:,0:-1])
#b_acc = sklearn.metrics.balanced_accuracy_score(B[1].iloc[:,-1], y_pred)  # equivalent to ROC_AUC in binary case
#acc = sklearn.metrics.accuracy_score(B[1].iloc[:,-1], y_pred)
#print("Test batch 1 - Balanced accuracy %f - Accuracy %f\n" % (b_acc, acc))
