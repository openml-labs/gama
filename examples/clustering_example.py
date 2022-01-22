from sklearn.datasets import load_breast_cancer
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from gama import GamaCluster

if __name__ == "__main__":

    X, y = load_breast_cancer(return_X_y=True)

    automl = GamaCluster(max_total_time=180, store="nothing", n_jobs=1)
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit(X)

    label_predictions = automl.predict(X)

    print("AMI:", adjusted_mutual_info_score(y, label_predictions))
    print("ARI:", adjusted_rand_score(y, label_predictions))
    print("Calinski-Harabasz:", calinski_harabasz_score(X, label_predictions))