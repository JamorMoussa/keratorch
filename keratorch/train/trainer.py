from ..state import ktState
from ..callbacks import CallBackList


from abc import ABC, abstractmethod


__all__ = ["ktTrainer", ]


class ktTrainer(ABC):

    def __init__(self):
        super(ktTrainer, self).__init__()

        self.state: ktState = ktState()
        self.callbacks = CallBackList(state=self.state)

    def compile(
        self, 
        model: 'kt.nn.ktModule', 
        optimizer: 'kt.optim.Optimizer', 
        loss_fn: 'Loss', 
        metrics: 'List[Metric]' = [],
        callbacks: 'List[Callback]' = []
    ):
        self.state.update(
            model = model
        )

    def train(
        self, 
        trainloader: 'DataLoader', 
        epochs: int, 
        *, 
        callbacks: 'List[Callback]'= [] 
    ):
        raise NotImplementedError


    def evaluate(
        self, 
        testloader: 'DataLoader'
    ):
        raise NotImplementedError