from gama import GamaCluster
from sklearn.datasets import load_breast_cancer
from gama.search_methods import AsynchronousSuccessiveHalving, RandomSearch, AsyncEA

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    automl = GamaCluster(max_total_time=180, store='all', n_jobs=1, scoring='normalized_mutual_info_score')
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

