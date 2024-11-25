import torch

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .callbacks.history import History
    from .utils.iters import TqdmIterator
    from torch.optim.optimizer import Optimizer


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


class ValidationState:
    pass


class TrainingState:

    def __init__(self):
        
        self.loss: float = None 
        self.batch: tuple[torch.Tensor] = None
        self.outputs: torch.Tensor = None 
        self.optimizer: "Optimizer" = None 

    def set_loss(self, loss: float):
        self.loss = loss

    def set_batch(self, batch: tuple[torch.Tensor]):
        self.batch = batch

    def set_outputs(self, outputs: torch.Tensor):
        self.outputs = outputs

    def set_optimizer(self, optimizer: "Optimizer"):
        self.optimizer = optimizer


class State:

    def __init__(
        self
    ):        
        self.model: torch.nn.Module = None 

        self.history: "History" = None 
        self.tqdm_iter: "TqdmIterator" = None 
        self.record_flag: bool = False

        self.hyparams = HyparamState()
        self.train = TrainingState()
        self.val = ValidationState() 

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def set_history(self, history: "History"):
        self.history = history

    def set_tqdm_iter(self, tqdm_iter: "TqdmIterator"):
        self.tqdm_iter = tqdm_iter
