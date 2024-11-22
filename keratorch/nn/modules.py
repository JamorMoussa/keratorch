import torch as tr, torch.nn as nn
from torch.utils.data import DataLoader


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..optim import Optimizer
from ..callbacks import CallBackList, CallBack , State, History
from ..utils.iters import TqdmIterator

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
        self.tqdm_iter = TqdmIterator()

        self.state = State()
        self.init_state()

        self.callbacklist: CallBackList = CallBackList(state=self.state)

        self.callbacklist.append(self.history)

    def init_state(self):
        self.state.set_model(model=self)
        self.state.set_optimizer(optimizer=self.optimzer)
        self.state.set_history(history=self.history)
        self.state.set_tqdm_iter(tqdm_iter=self.tqdm_iter)

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
        self, trainloader: DataLoader, num_iters: int, verbose_iter: int = None
    ):
        logs = {}

        self.state.hyprams.set_numiters(num_iters=num_iters)
        self.state.hyprams.set_loadersize(loadersize=len(trainloader))
        self.state.hyprams.set_verbose_iter(verbose_iter=verbose_iter)
        self.callbacklist.on_train_begin()

        for epoch in range(num_iters):
            
            epoch_loss = 0.0
            
            self.callbacklist.on_epoch_begin()

            for itr, batch in self.tqdm_iter.get_tqdm(loader=trainloader, enum=True):
                self.state.hyprams.set_epoch(epoch=epoch)
                self.state.hyprams.set_iter(iter=itr)

                self.state.set_batch(batch=batch)
                self.callbacklist.on_batch_begin()

                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)

                outputs = self.forward(inputs)

                self.state.set_outputs(outputs=outputs)
                
                loss: tr.Tensor = self.loss_fn(input=outputs, target=targets)

                self.state.set_loss(loss=loss.item())
                
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


