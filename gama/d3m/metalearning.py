from collections import Counter
import numpy as np


def calculate_meta_features(X, y):
    n_instances, n_features = X.shape
    dimensionality = n_features / n_instances

    class_counter = Counter(y)
    minority_class_size = class_counter.most_common()[-1][1]
    minority_class_percentage = (minority_class_size / len(y)) * 100
    majority_class_size = class_counter.most_common()[0][1]
    majority_class_percentage = (majority_class_size / len(y)) * 100
    n_classes = len(class_counter)

    n_missing_values = np.isnan(X).sum()
    perc_missing_values = (n_missing_values / (n_instances * n_features)) * 100
    n_instance_missing_values = np.isnan(X).sum(axis=1)
    perc_instance_missing_values = (n_instance_missing_values / n_instances) * 100

    auto_correlation = 1 - np.mean([int(y2 != y1) for y1, y2 in zip(y, y[1:])])
    return [
        n_features,
        auto_correlation,
        minority_class_size,
        n_missing_values,
        dimensionality,
        perc_instance_missing_values,
        majority_class_percentage,
        perc_missing_values,
        n_instance_missing_values,
        n_classes,
        n_instances,
        majority_class_size,
        minority_class_percentage
    ]


def SVC_features(individual):
    pass

def GradientBoosting_features(individual):
    pass

def RandomForestClassifier_features(individual):
    pass

def pick_best(metafeatures, configs, model, n):
    pass

def warm_start(X, y, n_each=5):
    metafeatures = calculate_meta_features(X, y)

    rfc_inviduals = [generate_rfc() for _ in range(100)]
    rfc_configs = [RandomForestClassifier_features(ind) for ind in rfc_inviduals]
    pick_best(metafeatures, rfc_configs)
