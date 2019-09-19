import math
from typing import Union, Generator

import pandas as pd

from gama.utilities.metrics import Metric


def prequential_sample(
        x: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        initial_batch_size: Union[int, float] = 0.5,
        batch_size: Union[int, float] = 0.1,
        n_batches: int = 5,
        train_include_all: bool = False
) -> Generator:
    """ Divide a chronological dataset up for prequential evaluation.

    Divides the dataset in the following fashion:
     - Create an initial batch of training data of size `initial_batch_size`.
     - Create `n_batches` of `batch_size` for evaluation.

    Assumes the data is already ordered chronologically.

    Parameters
    ----------
    x: pd.DataFrame
        features of the dataset
    y: pd.Series or pd.DataFrame
        target for each sample in x
    initial_batch_size: int or float (default=0.5)
        fraction (if float) or number (if int) of samples to include in the initial training batch.
    batch_size: int or float (default=0.5)
        fraction (if float) or number (if int) of samples to include in the subsequent train/test batches.
    n_batches: int (default=5)
        number of test batches to create
    train_include_all: bool (default=False)
        Include all data up to the test set for each train set. Convenient of partial training is not available.
        For example, a dataset of 100 samples to have 5 test sets of 10 samples each, the indices for training would be:
        If False: [0, 50], [50, 60], [60, 70], [70, 80], [80, 90]
        If True: [0, 50], [0, 60], [0, 70], [0, 80], [0, 90]


    Returns
    -------
    Generator
        Generates a sequence of tuples ((train_x, train_y), (test_x, test_y)).

    Raises
    ------
    ValueError
        If the splits would require more data than available.

    """
    if (isinstance(initial_batch_size, float)
            and isinstance(batch_size, float)
            and initial_batch_size + n_batches * batch_size > 1):
        raise ValueError("When `initial_batch_size` and `batch_size` are specified as float,"
                         "'initial_batch_size + n_batches * batch_size' may not exceed 1.")
    if (isinstance(initial_batch_size, int)
            and isinstance(batch_size, int)
            and initial_batch_size + n_batches * batch_size > len(y)):
        raise ValueError("When `initial_batch_size` and `batch_size` are specified as int,"
                         "'initial_batch_size + n_batches * batch_size' may not exceed len(y).")

    # Flooring because it is safe w.r.t. OutOfRange. Ignoring rounding errors for now (some samples might be dropped).
    if isinstance(batch_size, float):
        batch_size = math.floor(len(y) * batch_size)
    if isinstance(initial_batch_size, float):
        initial_batch_size = math.floor(len(y) * initial_batch_size)

    batch_indices = [(0, initial_batch_size)]
    for i in range(n_batches):
        batch_start_index = initial_batch_size + i * batch_size
        batch_indices.append((batch_start_index, batch_start_index + batch_size))

    for (train_start, train_end, test_start, test_end) in zip(batch_indices, batch_indices[1:]):
        if train_include_all:
            train_start = 0
        yield ((x.iloc[train_start:train_end], y.iloc[train_start:train_end]),
               (x.iloc[test_start:test_end], y.iloc[test_start:test_end]))


def prequential_score_predict(pipeline, x, y, metric: Metric):
    for ((x_train, y_train), (x_test, y_test)) in prequential_sample(x, y, train_include_all=True):
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        score = metric.maximizable_score(y_test, y_pred)
    pass
