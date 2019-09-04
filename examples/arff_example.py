from gama import GamaClassifier

if __name__ == '__main__':
    print("Make sure you adjust the file path if not executed from the examples directory.")
    file_path = "../tests/data/breast_cancer_{}.arff"

    automl = GamaClassifier(max_total_time=180, keep_analysis_log=False, n_jobs=1)
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit_arff(file_path.format('train'))

    label_predictions = automl.predict_arff(file_path.format('test'))
    probability_predictions = automl.predict_proba_arff(file_path.format('test'))
