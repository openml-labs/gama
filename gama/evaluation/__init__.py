import logging
import os
import pickle
from typing import List
import uuid

import pandas as pd
import stopit

from gama.utilities.metrics import Metric

log = logging.getLogger(__name__)


def score_pipeline(pipeline, x, y, evaluation_samples, draw_subsample, metrics: List[Metric]):
    """

    Parameters
    ----------
    pipeline:

    x:
        features
    y:
        targets
    evaluation_samples:
        pairs of ((x_train, y_train), (x_test, y_test)) to respectively call `train` and `predict` on
    draw_subsample:
        Method that draws a subset of (x, y) to perform evaluation on only a fraction of the data, if desired.
    metrics:
        Metrics to score the pipeline for.

    Returns
    -------

    """
    y_pred_concat = pd.Series()
    y_test_concat = pd.Series()

    x, y = draw_subsample(x, y)

    for ((x_train, y_train), (x_test, y_test)) in evaluation_samples(x, y):
        pipeline.fit(x_train, y_train)  # or partial_fit
        y_pred = pd.Series(pipeline.predict(x_test))  # or predict_proba
        y_test_concat = pd.concat([y_test_concat, y_test])
        y_pred_concat = pd.concat([y_pred_concat, y_pred])

    scores = [metric.maximizable_score(y_test_concat, y_pred_concat) for metric in metrics]

    return y_pred_concat, scores


def evaluate_pipeline(pipeline, x, y, evaluation_samples, draw_subsample, metrics, cache_dir, timeout, logger):
    if not logger:
        logger = log

    with stopit.ThreadingTimeout(timeout) as c_mgr:
        predictions, scores = score_pipeline(pipeline, x, y, evaluation_samples, draw_subsample, metrics)

    if cache_dir and -float("inf") not in scores and not draw_subsample:
        pl_filename = str(uuid.uuid4())
        try:
            with open(os.path.join(cache_dir, pl_filename + '.pkl'), 'wb') as fh:
                pickle.dump((pipeline, predictions, scores), fh)
        except FileNotFoundError:
            log.debug("File not found while saving predictions. This can happen in the multi-process case if the "
                      "cache gets deleted within `max_eval_time` of the end of the search process.", exc_info=True)
