import torch, torch.nn as nn
from torch.nn.modules.loss import _Loss

from typing import Any
from abc import ABC

__all__ = [
    "Loss", "MSELoss", "CrossEntropyLoss"
]


class ktBaseLoss(ABC):
    ... 


class Loss(_Loss, ktBaseLoss):

    _sub_class_loss: bool = True

    def __init__(
        self,
        torch_loss: _Loss = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean"
    ):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

        self.torch_loss = torch_loss

    def __call__(self, *args, **kwargs):
        args, kwargs = self._transform_args(*args, **kwargs)
        if self.torch_loss:
            return self.torch_loss(*args, **kwargs)
        
        return super().__call__(*args, **kwargs)

    def _transform_args(self, *args, **kwargs):
        return args, kwargs


class MSELoss(Loss):

    __doc__ == nn.MSELoss.__doc__

    def __init__(
        self,
        size_average: Any | None = None,
        reduce: Any | None = None,
        reduction: str = "mean"    
    ):
        super().__init__(
            torch_loss= nn.MSELoss(
                size_average = size_average,
                reduce = reduce,
                reduction = reduction,
            )
        )


class CrossEntropyLoss(Loss):

    __doc__ == nn.CrossEntropyLoss.__doc__

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average: Any | None = None,
        ignore_index: int = -100,
        reduce: Any | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0
    ):
        super().__init__(
            torch_loss= nn.CrossEntropyLoss(
                weight = weight,
                size_average = size_average,
                ignore_index = ignore_index,
                reduce = reduce,
                reduction = reduction,
                label_smoothing = label_smoothing,
            )
        )

    def _transform_args(
        self, target: torch.Tensor, input: torch.Tensor, *args, **kwargs
    ):
        target = target.flatten().long()
        return  (*args,), dict(input=input, target=target, **kwargs)