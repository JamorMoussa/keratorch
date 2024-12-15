import keratorch as kt

import torch, torch.nn as nn


class Trainer(kt.train.ktTrainer):

    def do_forward_pass(
        self, state: kt.state.State 
    ):
        state.train.batch


class Model(kt.nn.ktModule):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor):
        return self.fc(x)
    

model = Model()

trainer = kt.train.ktTrainer()

trainer.compile(
    model= model,
    optimizer = kt.optim.Adam(),
    loss_fn= nn.MSELoss(),
    metrics = [
        kt.metrics.Loss(name="loss"),
        kt.metrics.Accuracy(name="acc")
    ],
    callbacks = [
        kt.callbacks.TensorBord()
    ]
)

hist = trainer.train(
    trainloader= None,
    epochs= 10, 
)

trainer.evaluate(
    testloader= None 
)