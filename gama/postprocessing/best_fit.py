from typing import List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.base import TransformerMixin

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.utilities.export import (
    imports_and_steps_for_individual,
    transformers_to_str,
    format_import,
    format_pipeline,
)


class BestFitPostProcessing(BasePostProcessing):
    """ Post processing technique which trains the best found single pipeline. """

    def __init__(self, time_fraction: float = 0.1):
        super().__init__(time_fraction)
        self._selected_individual: Optional[Individual] = None

    def post_process(
        self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]
    ) -> object:
        self._selected_individual = selection[0]
        return self._selected_individual.pipeline.fit(x, y)

    def to_code(
        self, preprocessing: Sequence[Tuple[str, TransformerMixin]] = None
    ) -> str:
        if self._selected_individual is None:
            raise RuntimeError("`to_code` can only be called after `post_process`.")

        imports, steps = imports_and_steps_for_individual(self._selected_individual)
        if preprocessing is not None:
            trans_strs = transformers_to_str([t for _, t in preprocessing])
            names = [name for name, _ in preprocessing]
            steps = list(zip(names, trans_strs)) + steps
            imports = imports.union({format_import(t) for _, t in preprocessing})

        pipeline_statement = format_pipeline(steps)
        script = "\n".join(imports) + "\n\n" + pipeline_statement
        return script
