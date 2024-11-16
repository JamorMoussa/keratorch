from torch import optim 

from abc import ABC, abstractmethod

__all__ = ["Adam", "SGD", "O"]


class Optimizer(ABC):
    
    params: "optim.optimizer.ParamsT" = None

    def __init__(
        self, params: "optim.optimizer.ParamsT" = None, *args, **kwargs
    ):
        self.args = args 
        self.kwargs = kwargs

    def set_params(self, params: "optim.optimizer.ParamsT" = None):
        self.params = params
        self.super_init()

    @abstractmethod
    def super_init(self):
        ... 


class Adam(Optimizer, optim.Adam):

    def super_init(self):
        optim.Adam.__init__(self, self.params, *self.args, **self.kwargs)


class SGD(Optimizer, optim.SGD):

    def super_init(self):
        optim.SGD.__init__(self, self.params, *self.args, **self.kwargs)