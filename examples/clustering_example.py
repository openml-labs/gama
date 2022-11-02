import pandas as pd
from gama import GamaCluster
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":

    df = pd.read_csv('cc18/iris.csv')
    le = LabelEncoder()
    y = df.iloc[:,-1]
    y = le.fit_transform(y)
    X = df.iloc[:,:-1]

    automl=GamaCluster(max_total_time=180, store='all', n_jobs=1, scoring='normalized_mutual_info_score')
    print("Starting `fit` GamaCluster which will take roughly 3 minutes.")
    automl.fit(X, y)
    labels = automl.predict(X)
    #if external metric provided
    print(automl.score(y))
    #if internal metric provided
    #print(automl.score(X))
    print(automl.model)
    print(y)
    print(labels)

