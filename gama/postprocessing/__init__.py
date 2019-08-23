from gama.postprocessing.base_post_processing import BasePostProcessing
from gama.postprocessing.best_fit import BestFitPostProcessing
from gama.postprocessing.ensemble import EnsemblePostProcessing


class NoPostProcessing(BasePostProcessing):
    """ Dummy post-processing which does nothing. """
    def __init__(self, time_fraction: float = 0.0):
        super().__init__(time_fraction)

    def post_process(self, *args, **kwargs):
        return None


__all__ = ['NoPostProcessing', 'BestFitPostProcessing', 'EnsemblePostProcessing']
