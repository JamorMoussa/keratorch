from keras import Model
from keras.layers import TorchModuleWrapper

import torch.nn as nn


__all__ = ["ktModule", "build_model_from"]


class ktModule(Model):

    def __init__(self, torch_module, **kwargs):
        super().__init__(**kwargs)

        self.torch_module = TorchModuleWrapper(torch_module)

    def call(self, *args, **kwargs):
        return self.torch_module(*args, **kwargs)

def build_model_from(
    torch_module: nn.Module, 
    input_shape: tuple = None     
) -> ktModule: 
    
    model = ktModule(torch_module=torch_module)

    if input_shape:
        model.build(input_shape)

    return model 




    



    