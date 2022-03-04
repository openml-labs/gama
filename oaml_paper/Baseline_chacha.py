# Chacha online learning
#!pip install flaml[vw]
import pandas as pd
import numpy as np
import arff

from sklearn.metrics import accuracy_score
from vowpalwabbit import pyvw
from flaml import AutoVW
from river import metrics
import string
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

           ]

print(sys.argv[0]) # prints python_script.py
d = int(sys.argv[1])
print(f"Data stream is {datasets[d]}.")                      # prints dataset no

# Retrieve and prepare data
data_loc = datasets[d]      # needs to be arff
# Data
B = pd.DataFrame(arff.load(open(data_loc, 'r'), encode_nominal=True)["data"])

# Preprocessing of data: Drop NaNs, move target to the end, check for zero values

if d in [2, 3]:
    columns = B.columns.values.tolist()
    columns.remove(0)
    columns.append(0)
    B = B.reindex(columns, axis=1)

if pd.isnull(B.iloc[:, :]).any().any():
    print("Data X contains NaN values. The rows that contain NaN values will be dropped.")
    B.dropna(inplace=True)

if B[:].iloc[:, 0:-1].eq(0).any().any():
    print("Data contains zero values. They are not removed but might cause issues with some River learners.")

X = B[:].iloc[:, 0:-1]
y = B[:].iloc[:, -1]

#################################
# Convert into vowpalwabbit examples:
NS_LIST = list(string.ascii_lowercase) + list(string.ascii_uppercase)
max_ns_num = 10  # the maximum number of namespaces
orginal_dim = X.shape[1]
max_size_per_group = int(np.ceil(orginal_dim / float(max_ns_num)))
# sequential grouping
group_indexes = []
for i in range(max_ns_num):
    indexes = [ind for ind in range(i * max_size_per_group,
                                    min((i + 1) * max_size_per_group, orginal_dim))]
    if len(indexes) > 0:
        group_indexes.append(indexes)

vw_examples = []
for i in range(X.shape[0]):
    ns_content = []
    for zz in range(len(group_indexes)):
        ns_features = ' '.join('{}:{:.6f}'.format(ind, X.iloc[i, ind]) for ind in group_indexes[zz])
        ns_content.append(ns_features)
    ns_line = '{} |{}'.format(str(y[i]),
                              '|'.join('{} {}'.format(NS_LIST[j], ns_content[j]) for j in range(len(group_indexes))))
    vw_examples.append(ns_line)

###################################

max_iter_num = len(vw_examples)
online_metric = metrics.Accuracy()

# setup autoVW
autovw_ni = AutoVW(max_live_model_num=5, search_space={'interactions': AutoVW.AUTOMATIC, 'quiet': ''})

# online learning
for i in range(max_iter_num):
    vw_x = vw_examples[i]
    y_true = float(vw_examples[i].split('|')[0])
    # predict step
    y_pred = autovw_ni.predict(vw_x)
    # update online metric
    online_metric = online_metric.update(int(y_true), round(y_pred))
    # learn step
    autovw_ni.learn(vw_x)

    if i % 1000 == 0:
        print(f'Test batch - {i} with {online_metric}')
