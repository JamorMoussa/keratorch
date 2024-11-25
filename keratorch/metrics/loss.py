from .metric import Metric
from ..state import State

__all__ = ["Loss", ]

class Loss(Metric):

    def __init__(self, name: str ="loss"):
        super(Loss, self).__init__(name=name)

        self.counter: int = 1 

    def compute_train_metric(self, state):
        self.train_value += state.train.loss

    def compute_val_metric(self, state):
        self.val_value += state.val.loss
