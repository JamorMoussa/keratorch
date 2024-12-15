from dataclasses import dataclass, field

from .optim import ktOptimizer
from .nn import ktModule, ktLoss
from .utils.validations import TypeChecker

import torch


@dataclass
class HyparamState(TypeChecker):
    epochs: int = 1


@dataclass
class TrainState(TypeChecker):
    
    loss: torch.Tensor = field(default_factory= lambda: None)
    batch: tuple[torch.Tensor] = field(default_factory= lambda: (None, None))
    outputs: torch.Tensor = field(default_factory= lambda: None)


@dataclass
class ValidationState(TypeChecker):
    
    loss: torch.Tensor = field(default_factory= lambda: None)
    batch: tuple[torch.Tensor] = field(default_factory= lambda: (None, None))
    outputs: torch.Tensor = field(default_factory= lambda: None)

    do_validation: bool = False


@dataclass
class ktState(TypeChecker):

    model: ktModule = field(default_factory= lambda: None)
    optimizer: ktOptimizer = field(default_factory=lambda: None)
    loss_fn: ktLoss = field(default_factory=lambda: None)

    hyparams: HyparamState = field(default_factory=lambda: None)
    train: TrainState = field(default_factory=lambda: None)
    val: ValidationState = field(default_factory=lambda: None)

    def __post_init__(self):
        self.update(
            hyparams = HyparamState(),
            train = TrainState(),
            val = ValidationState()
        )


