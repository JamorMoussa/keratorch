from .  import TrModule

import torch.nn as nn
from torch.nn.modules.loss import _Loss


__all__ = ["Sequential", ]

class Sequential(nn.Sequential, TrModule):

    def __init__(
        self, *args: nn.Module
    ):
        super(Sequential, self).__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(module, _Loss):
                input = module(input)
        return input