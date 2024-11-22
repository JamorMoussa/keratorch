# Keratorch

**Keratorch** is a high-level API for PyTorch, inspired by Keras, aimed at simplifying the process of defining, compiling, and training models in PyTorch. Designed to enhance user experience, Keratorch enables developers to build neural networks in a modular and intuitive way, making PyTorch feel even more accessible.

## Key Features

- **Keras-like API**: Intuitive interface that follows the Keras workflow.
- **Flexible Model Architecture**: Easily define custom model layers and transformations.
- **Seamless Training Process**: Built-in methods for compiling and training models, reducing boilerplate code.

## Installation

To install Keratorch, clone the repository and install dependencies:

```bash
git clone https://github.com/JamorMoussa/keratorch.git
cd keratorch
pip install -r requirements.txt
```

## Getting Started

Here’s a quick example to get started with **Keratorch**.

### Import Packages

```python
import keratorch as kt

import torch as tr
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
```

### Model Definition

Use `Sequential`, a Keras-inspired container, to build the model. The `Lambda` layer allows custom transformations to be integrated easily.

```python
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
```

Create an instance of the `Model`: 

```python
model = Model()
```

### Model Compilation

Compile the model with a loss function and an optimizer.

```python
model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= kt.optim.Adam(lr=0.001),
    metrics= [
        kt.metrics.Loss(name="train loss"),
        # kt.metrics.Accuracy()
    ],
    callbacks= [ LogEpoch(), ]
)
```

### Data Preparation

Create synthetic data for training.

```python
x = tr.rand(1000, 7)
y = tr.mm(x, tr.rand(1, 7).t())

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=25)
```

### Training the Model

Train the model using the `fit` method.

```python
hist = model.fit(trainloader= loader, num_iters=5, num_records=40)
```

Training output:
```
Epoch: [0/5] | train loss: 0.3189:  91%|█████████████████████████████████████████▋    | 284/313 [00:01<00:00, 277.55it/s]
```

### Visualizing Training Loss

Plot the training loss to observe the model's learning progress.

```python
print(hist.history.keys())

plt.plot(hist.history["train loss"])
plt.show()
```

![Training Loss](https://raw.githubusercontent.com/JamorMoussa/images/refs/heads/main/src/keratorch/loss_plot.png)

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and send pull requests.

## License

<!-- This project is licensed under the MIT License. -->

## Acknowledgments

Thanks to the PyTorch and Keras communities for their inspiring frameworks and continued innovations.
