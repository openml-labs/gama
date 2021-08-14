from typing import List, Optional, Sequence, Tuple

import pandas as pd
from river.base import Transformer

from gama.genetic_programming.components import Individual
from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.utilities.export import (
    imports_and_steps_for_individual,
    transformers_to_str,
    format_import,
    format_pipeline,
)
from river import compose

class BestFitOnlinePostProcessing(BasePostProcessing):
    """ Post processing technique which trains the best found single pipeline. """

    def __init__(self, time_fraction: float = 0.1):
        super().__init__(time_fraction)
        self._selected_individual: Optional[Individual] = None

    def post_process(
        self, x: pd.DataFrame, y: pd.Series, timeout: float, selection: List[Individual]
    ) -> object:
        self._selected_individual = selection[0]

        #TO BE CHANGED
        steps = list(self._selected_individual.pipeline.steps.values())
        for i in range(len(steps[0])):
            if i == 0:
                river_model = steps[0][i][1]
            else:
                river_model |= steps[0][i][1]

        #print(river_model)
        # final = next(steps)
        # river_model = compose.Pipeline(final[0][1])
        for i in range(0, len(x)):
            river_model.learn_one(x.iloc[i].to_dict(), int(y[i]))

        #self._selected_individual.pipeline = river_model

        return river_model

    def to_code(
        self, preprocessing: Sequence[Tuple[str, Transformer]] = None
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
