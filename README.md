# keratorch
Keratorch: A Keras-style high-level API for building and training models in PyTorch


```python
import torch as tr, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import keratorch as kt

import matplotlib.pyplot as plt


```

```python

class MLPModel(kt.nn.TrModule):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(3, 1)

    def forward(self, x: tr.Tensor):
        return self.fc(x)


```

```python

model = MLPModel()


model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= tr.optim.Adam(model.parameters(), lr=0.01), 
)

```

```python
x = tr.rand(1000, 3)

y = tr.mm(
    x, tr.Tensor([[-1, 1, -1]]).t()
)

dataset = TensorDataset(x, y)

loader = DataLoader(dataset, batch_size=25)

```

```python
res = model.fit(trloader= loader, num_iters=10)
```

```
100%|█████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 19.10it/s]
```

```python
plt.plot(res["train_loss"])
plt.show()
```

![](https://raw.githubusercontent.com/JamorMoussa/images/refs/heads/main/src/keratorch/loss_plot.png)
