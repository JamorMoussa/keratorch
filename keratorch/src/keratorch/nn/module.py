from keras import Model
from keras.src.utils.torch_utils import TorchModuleWrapper
import torch, torch.nn as nn

from .loss import ktBaseLoss

class kerasModel(Model):
    def __init__(self, torch_module: nn.Module):
        super().__init__()
        self.torch_module = TorchModuleWrapper(torch_module)
        
    def call(self, *args, **kwargs):

        if args and kwargs:
            raise ValueError("Only positional arguments (*args) or keyword arguments (**kwargs) are allowed, not both.")

        if len(args) == 1:
            if type(args[0]) is dict:
                kwargs = args[0]
                args = ()

        return self.torch_module(*args, **kwargs)
    
    def compile(
        self, 
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        
        if not isinstance(loss, ktBaseLoss):
            raise ValueError(f"Expect `loss` to be a subclass of `kt.nn.Loss` class, but `{loss.__class__}` is given.")

        super().compile(
            optimizer = optimizer,
            loss = loss,
            loss_weights = loss_weights,
            metrics = metrics,
            weighted_metrics = weighted_metrics,
            run_eagerly = run_eagerly,
            steps_per_execution = steps_per_execution,
            jit_compile = jit_compile,
            auto_scale_loss = auto_scale_loss,
        )


def build_model_from(
    torch_module: nn.Module,
) -> kerasModel:
    
    return kerasModel(torch_module=torch_module)


