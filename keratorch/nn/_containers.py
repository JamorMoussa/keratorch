from .  import ktModule
from ..outs import ModelOutput

import torch.nn as nn

__all__ = ["ktSequential", ]

class ktSequential(nn.Sequential, ktModule):

    def __init__(
        self, *args: nn.Module
    ):
        super(ktSequential, self).__init__(*args)

    def forward(self, input):
        for module in self:
            input = module(input)
        return ModelOutput(outputs=input)