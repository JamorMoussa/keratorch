import torch as tr, torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable
from collections import defaultdict

from ..optim import Optimizer
from ..callback import CallBackList, CallBack , State, History

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss

__all__ = ["ktModule", "Lambda"]

class ktModule(nn.Module, ABC):

    def __init__(self):
        super(ktModule, self).__init__()

        self.loss_fn: "_Loss" = None 
        self.optimzer: "Optimizer" = None 
        self.device: tr.device = None

        self.history: History = History()

        self.state = State()
        self.init_state()

        self.callbacklist: CallBackList = CallBackList(state=self.state)

        self.callbacklist.append(self.history)

    def init_state(self):
        self.state.set_model(model=self)
        self.state.set_optimizer(optimizer=self.optimzer)
        self.state.set_history(history=self.history)
        # self.state.set_loss_fn()

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def compile(
        self, 
        loss_fn: "_Loss",
        optimizer: Optimizer,
        *,
        device: tr.device = None,
        callbacks: list[CallBack] = []
    ):
        self.loss_fn = loss_fn

        self.optimzer = optimizer
        self.optimzer.set_params(params=self.parameters())
        
        self.device = device if device is not None else tr.device("cpu")
        self.to(self.device)

        self.callbacklist.append(*callbacks)


    def fit(
        self, trloader: DataLoader, num_iters: int
    ):
        logs = {}

        self.callbacklist.on_train_begin()
        for epoch in range(num_iters):
            
            epoch_loss = 0.0
            
            self.callbacklist.on_epoch_begin()
            for itr, batch in enumerate(tqdm(trloader)):
                self.state.hyprams.set_epoch(epoch=epoch)
                self.state.hyprams.set_iter(iter=itr)

                self.state.set_batch(batch=batch)
                self.callbacklist.on_batch_begin()

                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outs_pred = self.forward(inputs)
                loss: tr.Tensor = self.loss_fn(input=outs_pred, target=targets)

                logs["train_loss"] = loss.item()
                
                self.optimzer.zero_grad()
                loss.backward()
                self.optimzer.step()

                self.state.set_logs(logs=logs)
                self.callbacklist.on_batch_end()

            self.callbacklist.on_epoch_end()

        return self.history

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


class Lambda(nn.Module):

    def __init__(
        self, transform_func: Callable
    ):
        super(Lambda, self).__init__()
        self.transform_func = transform_func

    def forward(self, inputs: tr.Tensor):
        return self.transform_func(inputs)