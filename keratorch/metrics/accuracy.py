from .metric import Metric
from ..callbacks import State

__all__ = ["Accuracy", ]


class Accuracy(Metric):

    def __init__(self, name: str = "acc"):
        super(Accuracy, self).__init__(name=name)

    def compute_value(self, state: State):

        if state.record_flag:
            self.metric_value /= state.hyprams.loadersize

        self.metric_value += (
            state.outputs.argmax(dim=-1) == state.batch[1]
        ).sum().item()

        self.metric_value /= self.counter 