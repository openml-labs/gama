from gama import GamaClassifier

if __name__ == '__main__':
    print("Make sure you adjust the file path if not executed from the examples directory.")
    file_path = "../tests/data/breast_cancer_{}.arff"

    automl = GamaClassifier(max_total_time=180)
    print("Starting `fit` which will take roughly 3 minutes.")
    automl.fit(arff_file_path=file_path.format('train'))

    label_predictions = automl.predict(arff_file_path=file_path.format('test'))
    probability_predictions = automl.predict_proba(arff_file_path=file_path.format('test'))
