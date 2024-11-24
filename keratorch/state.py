import torch

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .callbacks.history import History
    from .utils.iters import TqdmIterator


__all__ = ["State", ]

class HyparamState:
    iter: int = 0 
    epoch: int = 0
    num_records: int = 0
    loadersize: int = 0
    num_iters: int = 0
    bacth_size: int = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch 

    def set_iter(self, iter: int):
        self.iter = iter 

    def set_num_records(self, num_records: int):
        if num_records is None:
            num_records = 2 * self.num_iters
        self.num_records = num_records 

    def set_loadersize(self, loadersize: int):
        self.loadersize = loadersize

    def set_numiters(self, num_iters: int):
        self.num_iters = num_iters

    def set_batch_size(self, batch_size: int):
        self.bacth_size = batch_size


class State:

    model: torch.nn.Module = None 
    optimizer: "torch.optim.optimizer.Optimizer" = None 
    loss: float
    batch: tuple[torch.Tensor] = None 
    outputs: torch.Tensor = None
    history: "History" = None 
    tqdm_iter: "TqdmIterator" = None 
    record_flag: bool = False

    hyparams: HyparamState = HyparamState()

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def set_optimizer(self, optimizer: "torch.optim.optimizer.Optimizer"):
        self.optimizer = optimizer

    def set_loss(self, loss: float):
        self.loss = loss

    def set_batch(self, batch: tuple[torch.Tensor]):
        self.batch = batch

    def set_outputs(self, outputs: torch.Tensor):
        self.outputs = outputs

    def set_history(self, history: "History"):
        self.history = history

    def set_tqdm_iter(self, tqdm_iter: "TqdmIterator"):
        self.tqdm_iter = tqdm_iter
