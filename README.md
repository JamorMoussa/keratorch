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
            nn.ReLU(),
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

#### Make a Custom Metric:

You can make a custom metric:

```python
class PersonMetric(kt.metrics.Metric):

    def compute_metric(self, state: kt.state.State):
        
        outs = state.outputs.flatten()
        targets = state.batch[1].flatten()

        outs_ = outs - outs.mean()
        targets_ = targets - targets.mean()
        self.metric_value += (outs_ * targets_).sum().item() / ((outs_**2).sum().sqrt() * (targets_**2).sum().sqrt()).item()

```

### Model Compilation

Compile the model with a loss function and an optimizer.

```python
model.compile(
    loss_fn= nn.MSELoss(),
    optimizer= kt.optim.Adam(lr=0.001),
    metrics= [
        kt.metrics.Loss(name="loss"),
        PersonMetric(name="person")
        # kt.metrics.Accuracy()
    ],
    callbacks= []
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
Epoch: [0/10] | loss: 0.1505 | person: 0.6649: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 286.71it/s]
Epoch: [1/10] | loss: 0.0804 | person: 0.8177: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 290.66it/s]
Epoch: [2/10] | loss: 0.0702 | person: 0.8433: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 300.28it/s]
Epoch: [3/10] | loss: 0.0685 | person: 0.8481: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 309.34it/s]
Epoch: [4/10] | loss: 0.0683 | person: 0.8489: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 288.74it/s]
Epoch: [5/10] | loss: 0.0683 | person: 0.8491: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 287.12it/s]
Epoch: [6/10] | loss: 0.0684 | person: 0.8491: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 276.69it/s]
Epoch: [7/10] | loss: 0.0684 | person: 0.8492: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 247.33it/s]
Epoch: [8/10] | loss: 0.0684 | person: 0.8492: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 293.97it/s]
Epoch: [9/10] | loss: 0.0685 | person: 0.8493: 100%|██████████████████████████████████| 400/400 [00:01<00:00, 291.07it/s]
dict_keys(['loss', 'person'])
```

### Visualizing Training Loss

Plot the training loss to observe the model's learning progress.

```python
plt.plot(hist.history["loss"])
plt.show()
plt.plot(hist.history["person"])
plt.show()
```

![Training Loss](https://raw.githubusercontent.com/JamorMoussa/images/refs/heads/main/src/keratorch/loss_plot.png)

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and send pull requests.

## License

<!-- This project is licensed under the MIT License. -->

## Acknowledgments

Thanks to the PyTorch and Keras communities for their inspiring frameworks and continued innovations.
