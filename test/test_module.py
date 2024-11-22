import torch as tr, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import keratorch as kt

import matplotlib.pyplot as plt


class Model(kt.nn.ktModule):

    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Sequential(
            kt.nn.Lambda(lambda x: x.unsqueeze(1)),
            nn.Conv1d(1, 10, kernel_size=2),
            nn.Flatten(start_dim=1),
            nn.Linear(10 * 6 , 1)
        )

    def forward(self, x):
        return self.fc(x)


model = Model()


class LogEpoch(kt.callbacks.CallBack):

    def on_epoch_end(self, state = None):
        print("#"*30, state.hyprams.epoch, "#"*30)

model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= kt.optim.Adam(lr=0.001),
    metrics= [
        kt.metrics.Loss(name="train loss"),
        # kt.metrics.Accuracy()
    ],
    callbacks= [ LogEpoch(), ]
)

x = tr.rand(10000, 7)

y = tr.mm(
    x, tr.rand(1, 7).t()
) + 0.5* tr.randn(10000, 7)

print(y.shape)

dataset = TensorDataset(x, y)

loader = DataLoader(dataset, batch_size=32)

hist = model.fit(trainloader= loader, num_iters=5, num_records=40)

print(hist.history.keys())

plt.plot(hist.history["train loss"])
plt.show()



# iter == 0.1 * (len(loader)/batch_size )