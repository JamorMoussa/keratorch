from ..callbacks import CallBackList, CallBack , State, History
from ..utils.iters import TqdmIterator
from ..optim import Optimizer
from ..metrics import Metric

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch, torch.nn as nn


if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss


__all__ = ["ktTrainer", ]


class ktTrainer(nn.Module, ABC):

    def __init__(self):
        super(ktTrainer, self).__init__()

        self.history: History = History()

        self.tqdm_iter = TqdmIterator()

        self.state = State()

        self.callbacklist = CallBackList(
            state=self.state, history=self.history
        )

        self.loss_fn: "_Loss" = None 
        self.optimizer: "Optimizer" = None 
        self.device: torch.device = None

        self.initiate_states()


    def initiate_states(self):
        self.state.set_model(model=self)
        self.state.set_history(history=self.history)
        self.state.set_tqdm_iter(tqdm_iter=self.tqdm_iter)


    def compile_(self, *args, **kwargs):
        nn.Module.compile(self, *args, **kwargs)


    def compile(
        self, 
        loss_fn: "_Loss",
        optimizer: Optimizer,
        *,
        device: torch.device = None,
        metrics: list[Metric] = [],
        callbacks: list[CallBack] = [],
    ):
        self.callbacklist.clear()

        self.compile_optimizer_lossfn(
            optimizer=optimizer, loss_fn=loss_fn
        )

        self.send_model_to(device=device)

        self.callbacklist.append(*metrics)
        self.callbacklist.append(*callbacks)

        self.initiate_states()


    def compile_optimizer_lossfn(self, optimizer: Optimizer, loss_fn: "_Loss"):
        self.optimizer = optimizer
        self.optimizer.set_params(params=self.parameters())
        self.state.set_optimizer(optimizer=self.optimizer)

        self.loss_fn = loss_fn
    

    def send_model_to(
        self, device: torch.device
    ):
        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)


    def update_state_params_before_training(
        self, num_iters: int, loadersize: int, num_records: int, batch_size: int
    ):
        self.state.hyparams.set_numiters(num_iters=num_iters)
        self.state.hyparams.set_loadersize(loadersize=loadersize)
        self.state.hyparams.set_num_records(num_records=num_records)
        self.state.hyparams.set_batch_size(batch_size=batch_size)


    def update_state_params_after_iter(
        self, epoch: int, itr: int 
    ):
        self.state.hyparams.set_epoch(epoch=epoch)
        self.state.hyparams.set_iter(iter=itr)


    def do_forward_pass(
        self, batch: tuple[torch.Tensor]
    ):
        self.state.set_batch(batch=batch)

        inputs = batch[0].to(self.device)
        targets = batch[1].to(self.device)

        outputs = self.forward(inputs)
        self.state.set_outputs(outputs=outputs.detach())

        return outputs, targets


    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ):
        loss: torch.Tensor = self.loss_fn(input=outputs, target=targets)
        self.state.set_loss(loss=loss.item())

        return loss
    
    def do_backward_pass(
        self, loss: torch.Tensor
    ):        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        ... 

    @abstractmethod
    def fit(self, *args, **kwargs):
        ... 

    @abstractmethod
    def evaluate(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def summary(self):
        ...
