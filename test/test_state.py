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
    
trainer = kt.train.ktTrainer()

trainer.compile(
    model= Model(), 
    optimizer= kt.optim.Adam(lr=0.001), 
    loss_fn= nn.MSELoss(), 
    metrics= [
        kt.metrics.mse(),
        kt.metrics.person()
    ],
    callbacks= []
)

hist = trainer.train(
    trainloader=loader, epochs=50, num_records=200
)

plt.plot(hist.history["person"])
plt.show()

plt.plot(hist.history["mse"])
plt.show()


