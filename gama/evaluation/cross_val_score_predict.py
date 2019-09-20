from typing import Union
import pandas as pd

from gama.utilities.metrics import Metric

#
# def cross_val_predict_score(estimator, X, y_train, metrics=None, **kwargs):
#     """ Return both the predictions and score of the estimator trained on the data given the cv strategy.
#
#     :param y_train: target in appropriate format for training (typically (N,))
#     """
#     if not all(isinstance(metric, Metric) for metric in metrics):
#         raise ValueError('All `metrics` must be an instance of `metrics.Metric`, is {}.'
#                          .format([type(metric) for metric in metrics]))
#
#     predictions_are_probabilities = any(metric.requires_probabilities for metric in metrics)
#     method = 'predict_proba' if predictions_are_probabilities else 'predict'
#     predictions = cross_val_predict(estimator, X, y_train, method=method, **kwargs)
#
#     if predictions.ndim == 2 and predictions.shape[1] == 1:
#         predictions = predictions.squeeze()
#
#     scores = []
#     for metric in metrics:
#         if metric.requires_probabilities:
#             # `predictions` are of shape (N,K) and the ground truth should be formatted accordingly
#             y_ohe = OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()
#             scores.append(metric.maximizable_score(y_ohe, predictions))
#         elif predictions_are_probabilities:
#             # Metric requires no probabilities, but probabilities were predicted.
#             scores.append(metric.maximizable_score(y_train, predictions.argmax(axis=1)))
#         else:
#             # No metric requires probabilities, so `predictions` is an array of labels.
#             scores.append(metric.maximizable_score(y_train, predictions))
#
#     return predictions, scores
#
# def kfold_crossvalidation(x: pd.DataFrame, y: Union[pd.Series, pd.DataFrame])