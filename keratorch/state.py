from .optim import ktOptimizer
from .nn import ktModule, ktLoss
from .utils.validations import TypeChecker
from .outs import ModelOutput
from .callbacks import History
from .utils.iters import TqdmIterator

import torch

from dataclasses import dataclass, field
from typing import Any , TYPE_CHECKING




@dataclass
class HyparamState(TypeChecker):
    epoch: int = 0
    epochs: int = 0
    
    itr: int = 0

    num_records: int = 10
    record_flag: bool = False 


@dataclass
class MetricState(TypeChecker):

    metrics: dict = field(default_factory= lambda: {})
    tqdm_iter: TqdmIterator = field(default_factory= lambda: None)

    def update_metric(self, name: str, value: float):
        self.metrics[name] = value

        if self.tqdm_iter is not None: self.tqdm_iter.update_metrics()

    def items(self):
        return self.metrics.items()


@dataclass
class TrainState(TypeChecker):
    
    loss: torch.Tensor = field(default_factory= lambda: None)
    batch: list[torch.Tensor] = field(default_factory= lambda: [None, None])
    model_output: ModelOutput = field(default_factory= lambda: ModelOutput())

    loss: torch.Tensor = field(default_factory= lambda: None)

    loadersize: int = None 


@dataclass
class ValidationState(TypeChecker):
    
    loss: torch.Tensor = field(default_factory= lambda: None)
    batch: list[torch.Tensor] = field(default_factory= lambda: [None, None])
    model_output: ModelOutput = field(default_factory= lambda: ModelOutput())

    do_validation: bool = False


@dataclass
class ktState(TypeChecker):

    model: ktModule = field(default_factory= lambda: None)
    optimizer: ktOptimizer = field(default_factory=lambda: None)
    loss_fn: ktLoss | torch.nn.modules.loss._Loss = field(default_factory=lambda: None)
    device: torch.device = field(default_factory= lambda: torch.device("cpu"))

    hyparams: HyparamState = field(default_factory=lambda: None)
    train: TrainState = field(default_factory=lambda: None)
    val: ValidationState = field(default_factory=lambda: None)

    history: History = field(default_factory= lambda: None)
    metrics: MetricState = field(default_factory= lambda: None)
    tqdm_iter: TqdmIterator = field(default_factory= lambda: None)

    def __post_init__(self):
        self.update(
            hyparams = HyparamState(),
            train = TrainState(),
            val = ValidationState(),
            history = History(), 
            metrics = MetricState(),
        )


