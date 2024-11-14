import torch as tr, torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss
    from torch.optim.optimizer import Optimizer 

class TrModule(nn.Module, ABC):

    def __init__(self):
        super(TrModule, self).__init__()

        self.loss_fn: _Loss = None 
        self.optimzer: Optimizer = None 
        self.device: tr.device = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def compile(
        self, 
        loss_fn: _Loss,
        optimizer: Optimizer,
        *,
        device: tr.device = None,
    ):
        self.loss_fn = loss_fn
        self.optimzer = optimizer
        self.device = device if device is not None else tr.device("cpu")


    def fit(
        self, train_loader: DataLoader, num_iters: int
    ):
        results = {"train_loss": []} 

        for epoch in tqdm(range(num_iters)):

            for inputs, targets in train_loader:

                self.optimzer.zero_grad()

                outs_pred = self.forward(inputs.to(self.device))

                loss: tr.Tensor = self.loss_fn(input=outs_pred, target=targets.to(self.device))
                loss.backward()

                results["train_loss"].append(loss.item())

                self.optimzer.step()

        return results

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError