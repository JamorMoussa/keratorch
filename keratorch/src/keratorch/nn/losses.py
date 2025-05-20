import torch, torch.nn as nn
import torch.nn.functional as F

from typing import Any

__all__ = [
    "BaseLoss", "MSELoss", "CrossEntropyLoss"
]

class BaseLoss(nn.Module):

    def __init__(
        self, torch_loss
    ):
        super().__init__()
        self.torch_loss = torch_loss

    def forward(self, *args, **kwargs):
        args, kwargs = self._transform_args(*args, **kwargs)
        return self.torch_loss(*args, **kwargs)
    
    def _transform_args(self, *args, **kwargs):
        return args, kwargs 


class MSELoss(BaseLoss):

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


class CrossEntropyLoss(BaseLoss):

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

    def _transform_args(self, target: torch.Tensor, input: torch.Tensor, *args, **kwargs):
        
        target = target.flatten().long()

        return  (*args,), dict(input=input, target=target, **kwargs)