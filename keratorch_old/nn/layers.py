import torch, torch.nn as nn

from typing import Callable


__all__ = ["Lambda"]


class Lambda(nn.Module):

    def __init__(
        self, transform_func: Callable
    ):
        super(Lambda, self).__init__()
        self.transform_func = transform_func

    def forward(self, inputs: torch.Tensor):
        return self.transform_func(inputs)