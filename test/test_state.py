import keratorch as kt 

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt


x = torch.rand(10000, 2)
y = torch.mm(
    x, torch.Tensor([[1, 2]]).t()
) - 1

dataset = TensorDataset(x, y)

loader = DataLoader(dataset, batch_size=64, shuffle=True)


class Model(kt.nn.ktModule):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        
        return kt.ModelOutput(
            outputs= self.fc(x),
        )
    

@torch.no_grad()
def person(state: kt.state.ktState):

    outs = state.train.model_output.outputs.flatten()
    targets = state.train.batch[1].flatten()

    outs_ = outs - outs.mean()
    targets_ = targets - targets.mean()
    return (outs_ * targets_).sum().item() / ((outs_**2).sum().sqrt() * (targets_**2).sum().sqrt()).item()


@torch.no_grad()
def mse(state: kt.state.ktState):

    return F.mse_loss(
        state.train.model_output.outputs, state.train.batch[1]
    )


class Log(kt.callbacks.CallBack):

    def on_epoch_begin(self, state):
        print("#"*10, "Epoch: ", state.hyparams.epoch, "#"*30)

    def on_batch_end(self, state):
        
        if state.hyparams.record_flag:
            print("Loss: ", state.train.loss.item())


trainer = kt.train.ktTrainer()

trainer.compile(
    model= Model(), 
    optimizer= kt.optim.Adam(lr=0.001), 
    loss_fn= nn.MSELoss(), 
    callbacks= [
        kt.metrics.Metric(name="person", metric_func= person),
        kt.metrics.Metric(name="mse", metric_func= mse),
        Log(), 
    ]
)

hist = trainer.train(
    trainloader=loader, epochs=50, num_records=200
)

plt.plot(hist.history["person"])
plt.show()

plt.plot(hist.history["mse"])
plt.show()


