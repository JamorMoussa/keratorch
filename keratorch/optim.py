from torch import optim 

from abc import ABC, abstractmethod

__all__ = ["Adam", "SGD", "Optimizer"]


class Optimizer(ABC):
    
    @abstractmethod
    def set_params(self, params: "optim.optimizer.ParamsT" = None):
        ...


class Adam(optim.Adam, Optimizer):

    def __init__(
        self, params: "optim.optimizer.ParamsT" = None, *args, **kwargs
    ):
        self.args = args 
        self.kwargs = kwargs 

    def set_params(
        self, params: "optim.optimizer.ParamsT" = None,
    ):
        super(Adam, self).__init__(params=params, *self.args, **self.kwargs)


class SGD(optim.SGD, Optimizer):

    def __init__(
        self, params: "optim.optimizer.ParamsT" = None, *args, **kwargs
    ):
        self.args = args 
        self.kwargs = kwargs 

    def set_params(
        self, params: "optim.optimizer.ParamsT" = None,
    ):
        super(SGD, self).__init__(params=params, *self.args, **self.kwargs)