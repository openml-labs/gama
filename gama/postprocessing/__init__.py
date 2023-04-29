from typing import Optional, Sequence, Tuple

from sklearn.base import TransformerMixin

from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.postprocessing.best_fit import BestFitPostProcessing
from gama.postprocessing.ensemble import EnsemblePostProcessing


class NoPostProcessing(BasePostProcessing):
    """Does nothing, no time will be reserved for post-processing."""

    def to_code(
        self, preprocessing: Optional[Sequence[Tuple[str, TransformerMixin]]] = None
    ) -> str:
        raise NotImplementedError(
            "NoPostProcessing has no `to_code` function, since no model is selected."
        )

    def __init__(self, time_fraction: float = 0.0):
        super().__init__(time_fraction)

    def post_process(self, *args, **kwargs) -> None:
        return None


__all__ = ["NoPostProcessing", "BestFitPostProcessing", "EnsemblePostProcessing"]
