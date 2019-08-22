from abc import ABC
from typing import List, Dict, Tuple, Any

import pandas as pd

from gama.genetic_programming.operator_set import OperatorSet
from gama.genetic_programming.components import Individual


class BaseSearch(ABC):

    def __init__(self):
        # hyperparameters can be used to safe/process search hyperparameters
        self.hyperparameters: Dict[str, Tuple[Any, Any]] = dict()
        self.output: List[Individual] = []

    def dynamic_defaults(self, x: pd.DataFrame, y: pd.DataFrame, time: int):
        # updates self.hyperparameters defaults
        raise NotImplementedError("Must be implemented by child class.")

    def search(self, operations: OperatorSet, start_candidates: List[Individual]):
        raise NotImplementedError("Must be implemented by child class.")
