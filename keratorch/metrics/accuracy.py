from .metric import Metric
from ..callbacks import State

__all__ = ["Accuracy", ]


class Accuracy(Metric):

    def __init__(self, name: str = "acc"):
        super(Accuracy, self).__init__(name=name)

    def compute_metric(self, state: State):

        self.metric_value += (
            state.outputs.cpu().argmax(dim=-1) == state.batch[1]
        ).sum().item()

        if state.record_flag:
          self.metric_value /= state.hyparams.bacth_size

 