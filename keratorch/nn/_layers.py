import torch, torch.nn as nn

from typing import Callable


__all__ = ["ktLambda"]


class ktLambda(nn.Module):

    def __init__(
        self, transform_func: Callable[[torch.Tensor], torch.Tensor]
    ):
        super(ktLambda, self).__init__()
        self.transform_func = transform_func

    def forward(self, inputs: torch.Tensor):
        return self.transform_func(inputs)