from torch.nn.modules.loss import _Loss

from abc import ABC, abstractmethod


__all__ = ["ktLoss", ]

class ktLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction = "mean"):
        super().__init__(size_average, reduce, reduction)

    @abstractmethod
    def forward(self, *args, **kwargs):
        ... 