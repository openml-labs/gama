import copy
from typing import List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.utilities.export import individual_to_python


class BestFitPostProcessing(BasePostProcessing):
    """ Post processing technique which trains the best found single pipeline. """

    def __init__(self, time_fraction: float = 0.1):
        super().__init__(time_fraction)
        self._selected_individual = None

    def post_process(
        self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]
    ) -> "model":
        self._selected_individual = selection[0]
        return self._selected_individual.pipeline.fit(x, y)

    def to_code(self, preprocessing: Optional[Pipeline] = None) -> str:
        if preprocessing is not None:
            # We don't want to export the mapping of categorical encoders
            prepend = [
                (name, copy.copy(transformer))
                for name, transformer in preprocessing.steps
            ]
            for _, transformer in prepend:
                transformer.mapping = None
            return individual_to_python(self._selected_individual, prepend)
        return individual_to_python(self._selected_individual)
