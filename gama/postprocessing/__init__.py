from typing import List, Dict, Callable, Any

from .ensemble import build_fit_ensemble


class PostProcessing:

    def __init__(self,
                 name: str,
                 require: List[str],
                 arguments: Dict[str, Any],
                 time_fraction: float,
                 post_processing_function: Callable):
        """

        :param name: str
            a recognizable name
        :param require: List[str]
            Certain keywords that specify what to provide/cache
            Permitted 'require' keywords:
             - best: the best individual
             - cache: cache directory with predictions of all evaluated individuals
             - n_jobs: number of allowed parallel jobs
             - timeout: maximum allowed time in seconds
             - metric: the first metric for optimization
             - x: the training features
             - y: the training targets
             - encoder: the encoder/decoder for target labels (only if classification)
        :param arguments: Dict[str, Any]
            specifies arguments with which to call the post processing function
        :param time_fraction: float
            fraction of total time to reserve for post processing
        :param post_processing_function: Callable
            Callable with (**arguments)
        """
        self.name: str = name
        self.require: List[str] = require
        self.arguments: Dict[str, Any] = arguments
        self.time_fraction: float = time_fraction
        self.function_: Callable = post_processing_function


def fit_best(best, x, y):
    return best.pipeline.fit(x, y)


NoPostProcessing = PostProcessing(
    name='NoPostProcessing',
    require=[],
    arguments={},
    time_fraction=0,
    post_processing_function=lambda: None  # To satisfy type checker
)


FitBest = PostProcessing(
    name='FitBest',
    require=['best', 'x', 'y'],
    arguments={},
    time_fraction=0.1,
    post_processing_function=fit_best
)

Ensemble = PostProcessing(
    name='CaruanaEnsemble',
    require=['cache', 'x', 'y', 'n_jobs', 'timeout', 'metric', 'encoder'],
    arguments=dict(ensemble_size=25),
    time_fraction=0.3,
    post_processing_function=build_fit_ensemble
)


# We can not import GAMA as it would lead to circular import
def post_process(gama: 'Gama', method: PostProcessing, **kwargs) -> object:
    """ Perform post-processing after extracting required information from gama. """
    requires = dict(
        x=gama._X,
        y=gama._y,
        cache=gama._cache_dir,
        metric=gama._metrics[0],
        encoder=gama._label_encoder if hasattr(gama, '_label_encoder') else None
    )
    requires.update(method.arguments)
    requires.update(**kwargs)

    call_args = {k: v for (k, v) in requires.items()
                 if k in method.require or k in method.arguments}
    return method.function_(**call_args)
