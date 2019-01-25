from collections import Counter
import numpy as np
import pickle
import os

from ..genetic_programming.operations import create_seeded_individual


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
    n_instance_missing_values = sum(np.isnan(X).any(axis=1))
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
    svc_node = individual.main_node
    kernel_encoder = dict(poly=0, rbf=1, sigmoid=2)
    return [
        [t.value for t in svc_node._terminals if t.output == 'C'][0],
        [t.value for t in svc_node._terminals if t.output == 'coef0'][0],
        [t.value for t in svc_node._terminals if t.output == 'degree'][0],
        [t.value for t in svc_node._terminals if t.output == 'gamma'][0],
        kernel_encoder[[t.value for t in svc_node._terminals if t.output == 'kernel'][0]],
        [t.value for t in svc_node._terminals if t.output == 'shrinking'][0],
        [t.value for t in svc_node._terminals if t.output == 'tol'][0]
        ]


def GradientBoosting_features(individual):
    gbc_node = individual.main_node
    criterion_encoder = dict(friedman_mse=0, mae=1, mse=2)
    return [
        criterion_encoder[[t.value for t in gbc_node._terminals if t.output == 'criterion'][0]],
        [t.value for t in gbc_node._terminals if t.output == 'learning_rate'][0],
        [t.value for t in gbc_node._terminals if t.output == 'max_depth'][0],
        [t.value for t in gbc_node._terminals if t.output == 'max_features'][0],
        [t.value for t in gbc_node._terminals if t.output == 'min_impurity_decrease'][0],
        [t.value for t in gbc_node._terminals if t.output == 'min_samples_leaf'][0],
        [t.value for t in gbc_node._terminals if t.output == 'min_samples_split'][0],
        [t.value for t in gbc_node._terminals if t.output == 'min_weight_fraction_leaf'][0],
        [t.value for t in gbc_node._terminals if t.output == 'n_estimators'][0],
        [t.value for t in gbc_node._terminals if t.output == 'subsample'][0],
        [t.value for t in gbc_node._terminals if t.output == 'tol'][0],
        [t.value for t in gbc_node._terminals if t.output == 'validation_fraction'][0]
        ]


def RandomForestClassifier_features(individual):
    rfc_node = individual.main_node
    criterion_encoder = dict(gini=1, entropy=0)
    return [
        [t.value for t in rfc_node._terminals if t.output == 'bootstrap'][0],
        criterion_encoder[[t.value for t in rfc_node._terminals if t.output == 'criterion'][0]],
        [t.value for t in rfc_node._terminals if t.output == 'max_features'][0],
        [t.value for t in rfc_node._terminals if t.output == 'min_samples_leaf'][0],
        [t.value for t in rfc_node._terminals if t.output == 'min_samples_split'][0]
        ]


def pick_best(metafeatures, configs, model, n):
    combined_data = np.asarray([
        [*metafeatures, *config] for config in configs
    ])
    predicted_performance = model.predict(combined_data)
    return list(zip(*sorted(enumerate(predicted_performance), key=lambda x: x[1])[-n:]))[0]


def load_model(learner):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '{}.pkl'.format(learner))
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def generate_warm_start_pop(X, y, primitive_set, n_each=5):
    warm_start_pop = []
    metafeatures = calculate_meta_features(X, y)
    learners = [('SVC', SVC_features),
                ('RandomForestClassifier', RandomForestClassifier_features),
                ('GradientBoostingClassifier', GradientBoosting_features)]

    for learner, feature_fn in learners:
        model = load_model(learner)
        learner_prim = [p for p in primitive_set['prediction'] if learner in str(p)]
        if len(learner_prim) > 0:
            learner_prim = learner_prim[0]
        else:
            continue
        candidate_inds = [create_seeded_individual(primitive_set, learner_prim, max_length=1) for _ in range(100)]
        candidate_configs = [feature_fn(ind) for ind in candidate_inds]
        best_indices = pick_best(metafeatures, candidate_configs, model, n_each)
        warm_start_pop += [candidate_inds[i] for i in best_indices]

    return warm_start_pop
