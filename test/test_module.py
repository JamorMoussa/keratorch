import torch, torch.nn as nn
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

class PersonMetric(kt.metrics.Metric):


    def compute_person(self, outs, targets):

        outs = outs.flatten()
        targets = targets.flatten()

        outs_ = outs - outs.mean()
        targets_ = targets - targets.mean()
        return  (outs_ * targets_).sum().item() / ((outs_**2).sum().sqrt() * (targets_**2).sum().sqrt()).item()

    def compute_train_metric(self, state):
        self.train_value += self.compute_person(
            outs=state.train.outputs, targets=state.train.batch[1]
        )
    
    def compute_val_metric(self, state):
        self.val_value = self.compute_person(
            outs=state.val.outputs, targets=state.val.targets
        )

class LogEpoch(kt.callbacks.CallBack):

    def on_epoch_end(self, state = None):
        print("#"*30, state.hyparams.epoch, "#"*30)

model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= kt.optim.Adam(lr=0.01),
    metrics= [
        kt.metrics.Loss(name="loss"),
        # kt.metrics.Accuracy()
        PersonMetric(name="person")
    ],
    # callbacks= [ LogEpoch(), ]
)

x = torch.rand(10000, 7)

y = torch.mm(
    x, torch.rand(1, 7).t()
) + 0.2 * torch.randn(10000, 1)


x_test = torch.rand(1000, 7)

y_test = torch.mm(
    x_test, torch.rand(1, 7).t()
)


dataset = TensorDataset(x, y)

test_dataset = TensorDataset(x_test, y_test)

trainloader = DataLoader(dataset, batch_size=32, shuffle=True)

testloader = DataLoader(test_dataset, batch_size=32)

hist = model.fit(
    trainloader= trainloader, num_iters=20, num_records=100, val_split=0.2, val_records=100
)

eval_loss = model.evaluate(testloader=testloader)

print("#### Eval Loss:", eval_loss)

print(hist.history.keys())

plt.plot(hist.history["loss"])
plt.show()
plt.plot(hist.history["val_loss"])
plt.plot(hist.history["train_loss"])
plt.show()
plt.plot(hist.history["person"])
plt.show()
plt.plot(hist.history["val_person"])
plt.plot(hist.history["train_person"])
plt.show()



# iter == 0.1 * (len(loader)/batch_size )