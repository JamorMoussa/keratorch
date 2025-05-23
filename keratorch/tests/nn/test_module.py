import pytest
import torch.nn as nn
import torch
from keratorch.nn.module import build_model_from, kerasModel

def test_build_model_from():
    
    class DummyModule(nn.Module):
    
        def forward(self, x):
            return x * 2

    torch_module = DummyModule()

    model = build_model_from(torch_module)

    assert isinstance(model, kerasModel)

    assert model(torch.tensor([2.0])) == torch.tensor([4.0])

    assert model({"x": torch.tensor([2.0])}) == torch.tensor([4.0])

    assert model(x= torch.tensor([2.0])) == torch.tensor([4.0])