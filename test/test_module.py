import torch as tr, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import keratorch as kt

import matplotlib.pyplot as plt


# class MLPModel(kt.nn.TrModule):

#     def __init__(self):
#         super().__init__()

#         self.fc = nn.Linear(3, 1)

#     def forward(self, x: tr.Tensor):
#         return self.fc(x)

# model = MLPModel()



model = kt.nn.TrSequential(
    nn.Linear(3, 1)
)

print(model.__class__)


model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= tr.optim.Adam(model.parameters(), lr=0.01), 
)

x = tr.rand(1000, 3)

y = tr.mm(
    x, tr.Tensor([[-1, 1, -1]]).t()
)

dataset = TensorDataset(x, y)

loader = DataLoader(dataset, batch_size=25)

res = model.fit(trloader= loader, num_iters=10)


plt.plot(res["train_loss"])
plt.show()