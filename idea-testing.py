import numpy as np
import pandas as pd
import arff

#Data prep
a = arff.load(open('SEA_Abrubt_5.arff', 'r'),encode_nominal=True)
B = np.array_split(pd.DataFrame(a["data"]),500)
#B[0]
X = B[0].iloc[:,0:-1]
y = B[0].iloc[:,-1]
print(X,y)

from gama import GamaClassifier
from gama.search_methods import RandomSearch
from gama.postprocessing import NoPostProcessing

#Gama online - initial pipeline optimization

model = GamaClassifier(max_total_time=60,
                       scoring='accuracy',
                       search = RandomSearch(),
                       post_processing = NoPostProcessing())

model.fit(X,y)