import torch as tr, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import keratorch as kt

import matplotlib.pyplot as plt


class Model(kt.nn.ktModule):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = kt.nn.Sequential(
            kt.nn.Lambda(lambda x: x.unsqueeze(1)),
            nn.Conv1d(1, 10, kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(10 * 6 , 1)
        )

    def forward(self, x):
        return self.fc(x)


model = Model()

model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= kt.optim.Adam(lr=0.01),
    callbacks= []
)

x = tr.rand(1000, 7)

y = tr.mm(
    x, tr.rand(1, 7).t()
) + 0.3 * tr.randn(1000, 7)

dataset = TensorDataset(x, y)

loader = DataLoader(dataset, batch_size=25)

hist = model.fit(trloader= loader, num_iters=10)

plt.plot(hist.history["train_loss"])
plt.show()