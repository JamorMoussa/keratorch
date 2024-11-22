from ..callbacks import CallBackList, CallBack , State, History
from ..utils.iters import TqdmIterator
from ..optim import Optimizer
from ..metrics import Metric

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch as tr, torch.nn as nn


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
        self.device: tr.device = None

        self.initiate_states()


    def initiate_states(self):
        self.state.set_model(model=self)
        self.state.set_optimizer(optimizer=self.optimizer)
        self.state.set_history(history=self.history)
        self.state.set_tqdm_iter(tqdm_iter=self.tqdm_iter)


    def compile_(self, *args, **kwargs):
        nn.Module.compile(self, *args, **kwargs)


    def compile(
        self, 
        loss_fn: "_Loss",
        optimizer: Optimizer,
        *,
        device: tr.device = None,
        metrics: list[Metric] = [],
        callbacks: list[CallBack] = [],
    ):

        self.compile_optimizer_lossfn(
            optimizer=optimizer, loss_fn=loss_fn
        )

        self.send_model_to(device=device)

        self.callbacklist.append(*metrics)
        self.callbacklist.append(*callbacks)


    def compile_optimizer_lossfn(self, optimizer: Optimizer, loss_fn: "_Loss"):
        self.optimizer = optimizer
        self.optimizer.set_params(params=self.parameters())

        self.loss_fn = loss_fn
    

    def send_model_to(
        self, device: tr.device
    ):
        self.device = device if device is not None else tr.device("cpu")
        self.to(self.device)


    def update_state_params_before_training(
        self, num_iters: int, loadersize: int, num_records: int
    ):
        self.state.hyprams.set_numiters(num_iters=num_iters)
        self.state.hyprams.set_loadersize(loadersize=loadersize)
        self.state.hyprams.set_verbose_iter(verbose_iter=num_records)


    def update_state_params_after_iter(
        self, epoch: int, itr: int , batch: tuple[tr.Tensor]
    ):
        self.state.hyprams.set_epoch(epoch=epoch)
        self.state.hyprams.set_iter(iter=itr)


    def compute_forward(
        self, batch: tuple[tr.Tensor]
    ):
        self.state.set_batch(batch=batch)
        
        inputs = batch[0].to(self.device)
        targets = batch[1].to(self.device)

        outputs = self.forward(inputs)
        self.state.set_outputs(outputs=outputs)

        return outputs, targets


    def compute_loss(
        self, outputs: tr.Tensor, targets: tr.Tensor
    ):
        loss: tr.Tensor = self.loss_fn(input=outputs, target=targets)
        self.state.set_loss(loss=loss.item())

        return loss
    
    def do_backward_optimizer_step(
        self, loss: tr.Tensor
    ):        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
