from keras import Model as KerasModel
from keras.layers import TorchModuleWrapper

import torch.nn as nn


__all__ = ["ktModule", "build_from"]


class ktModule(KerasModel):

    def __init__(self, torch_module, **kwargs):
        super().__init__(**kwargs)

        self.torch_module = TorchModuleWrapper(torch_module)

    def call(self, *args, **kwargs):
        return self.torch_module(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.call(*args, **kwargs)
    
    def __repr__(self):
        return repr(self.torch_module.module)
    
    def __str__(self):
        return repr(self)


def build_from(
    torch_module: nn.Module, 
    input_shape: tuple = None     
) -> ktModule: 
    
    model = ktModule(torch_module=torch_module)

    if input_shape:
        model.build(input_shape)

    return model 




    



    