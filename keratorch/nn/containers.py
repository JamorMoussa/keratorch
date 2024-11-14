from .  import TrModule

from torch.nn import Sequential, Module
from torch.nn.modules.loss import _Loss


__all__ = ["TrSequential", ]

class TrSequential(Sequential, TrModule):

    def __init__(
        self, *args: Module
    ):
        super(TrSequential, self).__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(module, _Loss):
                input = module(input)
        return input