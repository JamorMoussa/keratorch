from .metric import Metric
from ..state import State

__all__ = ["Loss", ]

class Loss(Metric):

    def __init__(self, name: str ="loss"):
        super(Loss, self).__init__(name=name)

        self.counter: int = 1  

    def compute_metric(self, state: State):

        self.metric_value += state.loss
