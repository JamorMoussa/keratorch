import torch.nn as nn

from abc import ABC, abstractmethod


__all__ = ["ktModule", ]


class ktModule(nn.Module, ABC):

    def __init__(self):
        super(ktModule, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass 