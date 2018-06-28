from collections import Counter

import logging
import numpy as np

from .pretty_string_methods import clean_pipeline_string


log = logging.getLogger(__name__)


def optimal_constant_predictor(y_tr, metric):
    mean_is_best_for = ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_mean_squared_log_error", "mean_squared_error", "r2"]
    median_is_best_for = ["neg_median_absolute_error", "median_absolute_error"]

    # Constant predictions for averaged f1/recall/precision always lead to an invalid score since one will be 0.
    majority_class_is_best_for = ["roc_auc", "accuracy"]
    class_probabilities_best_for = ["log_loss", "neg_log_loss"]

    if metric.name in mean_is_best_for:
        return np.mean(y_tr)
    elif metric.name in median_is_best_for:
        return np.median(y_tr)
    elif metric.name in class_probabilities_best_for:
        if y.ndim == 1:
            return [count / len(y_tr) for class_, count in sorted(Counter(y_tr).most_common())]
        else:
            return np.mean(y_tr, axis=1)
    elif metric.name in majority_class_is_best_for:
        # Majority class is fallback option anyway.
        pass

    log.warning("No best constant predictor set for {}. Falling back on majority class.".format(metric.name))
    if y_tr.ndim == 1:
        y_tr_counter = Counter(y_tr)
        majority_class = y_tr_counter.most_common(1)[0][0]
        return [1 if idx == majority_class else 0 for idx in range(len(y_tr_counter))]
    else:
        return np.argmax(np.sum(y_tr, axis=1))
