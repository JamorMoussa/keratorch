import torch as tr

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .history import History


__all__ = ["State", ]

class HyParamState:
    iter: int = 0 
    epoch: int = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch 

    def set_iter(self, iter: int):
        self.iter = iter 


class State:

    model: tr.nn.Module = None 
    optimizer: "tr.optim.optimizer.Optimizer" = None 
    loss_fn: tr.nn.modules.loss._Loss = None
    batch: tuple[tr.Tensor] = None 
    history: "History" = None 
    logs: dict[Any] = {} 

    hyprams: HyParamState = HyParamState()

    def set_model(self, model: tr.nn.Module):
        self.model = model

    def set_optimizer(self, optimizer: "tr.optim.optimizer.Optimizer"):
        self.optimizer = optimizer

    def set_loss_fn(self, loss_fn: tr.nn.modules.loss._Loss):
        self.loss_fn = loss_fn

    def set_batch(self, batch: tuple[tr.Tensor]):
        self.batch = batch

    def set_history(self, history: "History"):
        self.history = history

    def set_logs(self, logs: dict[Any]):
        self.logs = logs