from .metric import Metric
from ..callbacks import State

import torch

__all__ = ["Accuracy", ]


class Accuracy(Metric):

    def __init__(self, name: str = "acc"):
        super(Accuracy, self).__init__(name=name)


    def compute_acc(self, outputs: torch.Tensor, targets: torch.Tensor):

        return (
            outputs.argmax(dim=-1) == targets
        ).sum().item()
    

    def compute_train_metric(self, state):
        self.train_value += self.compute_acc(
            outputs=state.train.outputs, targets= state.train.batch[1]
        )

        if state.record_flag:
          self.train_value /= state.hyparams.bacth_size

    def compute_val_metric(self, state):
        self.val_value = self.compute_acc(
            outputs=state.val.outputs, targets= state.val.targets
        )

        self.val_value /= state.val.outputs.size(0)



 