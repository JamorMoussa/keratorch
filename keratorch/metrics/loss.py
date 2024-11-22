from typing import Dict, Any, List
from collections import defaultdict

from .metric import Metric
from ..state import State

__all__ = ["Loss", ]

class Loss(Metric):

    loss: float 
    counter: int = 1  

    def __init__(self, name: str ="loss"):
        super(Loss, self).__init__(name=name)

    def compute_value(
        self, state: State
    ):
        self.metric_value += state.loss

        return self.metric_value
