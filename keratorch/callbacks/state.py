import torch as tr

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .history import History
    from ..utils.iters import TqdmIterator


__all__ = ["State", ]

class HyparamState:
    iter: int = 0 
    epoch: int = 0
    verbose_iter: int = 0
    loadersize: int = 0
    num_iters: int = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch 

    def set_iter(self, iter: int):
        self.iter = iter 

    def set_verbose_iter(self, verbose_iter: int):
        if verbose_iter is None:
            verbose_iter = 2 * self.num_iters
        self.verbose_iter = verbose_iter 

    def set_loadersize(self, loadersize: int):
        self.loadersize = loadersize

    def set_numiters(self, num_iters: int):
        self.num_iters = num_iters


class State:

    model: tr.nn.Module = None 
    optimizer: "tr.optim.optimizer.Optimizer" = None 
    loss: float
    batch: tuple[tr.Tensor] = None 
    outputs: tr.Tensor = None
    history: "History" = None 
    tqdm_iter: "TqdmIterator" = None 
    logs: dict[Any] = {}
    record_flag: bool = False

    hyprams: HyparamState = HyparamState()

    def set_model(self, model: tr.nn.Module):
        self.model = model

    def set_optimizer(self, optimizer: "tr.optim.optimizer.Optimizer"):
        self.optimizer = optimizer

    def set_loss(self, loss: float):
        self.loss = loss

    def set_batch(self, batch: tuple[tr.Tensor]):
        self.batch = batch

    def set_outputs(self, outputs: tr.Tensor):
        self.outputs = outputs

    def set_history(self, history: "History"):
        self.history = history

    def set_tqdm_iter(self, tqdm_iter: "TqdmIterator"):
        self.tqdm_iter = tqdm_iter

    def set_logs(self, logs: dict[Any]):
        self.logs = logs