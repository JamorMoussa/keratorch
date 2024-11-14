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

Use `TrSequential`, a Keras-inspired container, to build the model. The `TrLambda` layer allows custom transformations to be integrated easily.

```python
model = kt.nn.TrSequential(
    kt.nn.TrLambda(lambda x: x.unsqueeze(1)),  # Custom transformation
    nn.Conv1d(1, 10, kernel_size=2),
    nn.Flatten(start_dim=1),
    nn.Linear(10 * 6, 1)
)
```

### Model Compilation

Compile the model with a loss function and an optimizer.

```python
model.compile(
    loss_fn=nn.MSELoss(),
    optimizer=tr.optim.Adam(model.parameters(), lr=0.01),
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
res = model.fit(trloader=loader, num_iters=10)
```

Training output:
```
100%|█████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 19.10it/s]
```

### Visualizing Training Loss

Plot the training loss to observe the model's learning progress.

```python
plt.plot(res["train_loss"])
plt.show()
```

![Training Loss](https://raw.githubusercontent.com/JamorMoussa/images/refs/heads/main/src/keratorch/loss_plot.png)

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and send pull requests.

## License

<!-- This project is licensed under the MIT License. -->

## Acknowledgments

Thanks to the PyTorch and Keras communities for their inspiring frameworks and continued innovations.
