from typing import List

import pandas as pd

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing


class BestFitPostProcessing(BasePostProcessing):
    """ Post processing technique which trains the best found single pipeline. """

    def __init__(self, time_fraction: float = 0.1):
        super().__init__(time_fraction)

    def post_process(self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]) -> 'model':
        return selection[0].pipeline.fit(x, y)
