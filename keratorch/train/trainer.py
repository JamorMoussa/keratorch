from ..state import ktState
from ..callbacks import CallBackList
from ..outs import ModelOutput

import keratorch as kt

from typing import Any 
from abc import ABC

import torch 

__all__ = ["ktTrainer", ]


class ktTrainer(ABC):

    def __init__(self):
        super(ktTrainer, self).__init__()

        self.state: ktState = ktState()
        self.callbacks = CallBackList(state=self.state)

    def compile(
        self, 
        model: kt.nn.ktModule, 
        optimizer: kt.optim.ktOptimizer, 
        loss_fn: kt.nn.ktLoss | torch.nn.modules.loss._Loss, 
        *, 
        device: torch.device = torch.device("cpu"),
        metrics: 'List[Metric]' = [],
        callbacks: 'List[Callback]' = []
    ):
        self.state.update(
            model= model, optimizer= optimizer, loss_fn= loss_fn,  device= device
        )
        self.state.model.to(self.state.device)
        self.state.optimizer.set_params(
            params= self.state.model.parameters()
        )

        self.callbacks.train.append(*callbacks)

    def train(
        self, 
        trainloader: 'DataLoader', 
        epochs: int, 
        *, 
        callbacks: 'List[Callback]'= [] 
    ):
        self.state.model.train()
        self.callbacks.train.on_train_begin()

        for epoch in range(epochs):
            
            self.state.hyparams.update(epoch= epoch)
            
            self.callbacks.train.on_epoch_begin()

            for itr, batch in enumerate(trainloader):
            
                self.state.hyparams.update(itr= itr)
                self.state.train.update(batch= batch)

                self.callbacks.train.on_batch_begin()

                self.do_zero_grad()

                outs: ModelOutput = self.do_forward_pass(batch= batch)

                self.state.train.update(outputs= outs)
                self.check_outputs_type(outputs= outs)

                loss = self.compute_loss(outputs= outs.outputs, targets= batch[1])

                self.do_backward_pass(loss= loss)

                self.do_optimizer_step()

                self.callbacks.train.on_batch_end()

            self.callbacks.train.on_epoch_end()


    def check_outputs_type(self, outputs: ModelOutput | Any):
        if not isinstance(outputs, ModelOutput):
            raise TypeError(f"model's outputs must be of type 'ModelOutput', got '{type(outputs).__name__}' instead.")


    def do_zero_grad(self):
        self.state.optimizer.zero_grad()


    def do_forward_pass(
        self, batch: tuple[torch.Tensor]
    ) -> ModelOutput:
        inputs = batch[0].to(self.state.device)

        return self.state.model(inputs)


    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor 
    ): 
        loss =  self.state.loss_fn(outputs, targets) 
        self.state.train.update(loss= loss)

        return loss 
    

    def do_backward_pass(
        self, loss: torch.Tensor
    ):
        loss.backward() 


    def do_optimizer_step(self):
        self.state.optimizer.step()


    def evaluate(
        self, 
        testloader: 'DataLoader'
    ):
        raise NotImplementedError