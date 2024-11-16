import torch as tr, torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable
from collections import defaultdict

from ..optim import Optimizer
from ..callback import CallBackList, CallBack, History

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss

__all__ = ["TrModule", "Lambda"]

class TrModule(nn.Module, ABC):

    def __init__(self):
        super(TrModule, self).__init__()

        self.loss_fn: "_Loss" = None 
        self.optimzer: "Optimizer" = None 
        self.device: tr.device = None

        self.callbacklist: CallBackList = CallBackList()
        self.history: History = History()
        self.callbacklist.append(self.history)

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
            
            self.callbacklist.on_epoch_begin(epoch=epoch)
            for inputs, targets in tqdm(trloader):

                self.callbacklist.on_batch_begin(batch=[inputs, targets])

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimzer.zero_grad()

                outs_pred = self.forward(inputs)

                loss: tr.Tensor = self.loss_fn(input=outs_pred, target=targets)
                loss.backward()

                logs["train_loss"] = loss.item()
                
                self.optimzer.step()

                self.callbacklist.on_batch_end(batch=(inputs, targets), logs=logs)

            self.callbacklist.on_epoch_end(epoch=epoch, logs=logs)

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