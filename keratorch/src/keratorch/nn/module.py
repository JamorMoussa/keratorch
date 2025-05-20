from keras import Model
from keras.layers import TorchModuleWrapper
import torch, torch.nn as nn


class kerasModel(Model):
    def __init__(self, torch_module: nn.Module):
        super().__init__()
        self.torch_module = TorchModuleWrapper(torch_module)
        
    def call(self, *args, **kwargs):

        if len(args) == 1:
            kwargs = args[0]
            args = ()

        return self.torch_module(*args, **kwargs)


def build_model_from(
    torch_module: nn.Module,
) -> kerasModel:
    
    return kerasModel(torch_module=torch_module)


