<p align="center">
<img src="./asset/images/logo.png" width="300px">
</p>

# keratorch
Keratorch: A Keras-style high-level API for building and training models in PyTorch.


## Example:

```python
import torch 
import torch.nn as nn
import keratorch as kt


class UserEmbeddingModel(nn.Module):

    def __init__(self, num_users: int = 10, embed_dim: int = 8):
        super().__init__()

        self.user_embd = nn.Embedding(
            num_embeddings=num_users, embedding_dim= embed_dim
        )

        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, user_ids: torch.LongTensor):
        return self.user_embd(user_ids)


model = kt.nn.build_model_from(
    torch_module= UserEmbeddingModel(),
)

print(model.summary())

X = torch.randint(0, 10, (120,))

y = torch.mm(torch.rand(120, 10), torch.rand(10, 1)) - 1

trainloader = torch.utils.data.DataLoader(
    dataset= torch.utils.data.TensorDataset(X, y), batch_size=32
)

def torch_style_loss(y_true, y_pred):
    # y_true, y_pred are torch.Tensors already
    return torch.mean((y_pred - y_true)**2 + 0.01 * torch.abs(y_pred))


model.compile(
    loss= torch_style_loss,
    optimizer="adam",
    metrics=["mse", "mae"]
)
model.fit(trainloader, epochs=5)

```

```zsh
Model: "kt_module"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ torch_module_wrapper                 │ ?                           │              89 │
│ (TorchModuleWrapper)                 │                             │                 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 89 (356.00 B)
 Trainable params: 89 (356.00 B)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/5
/home/moussa/Documents/programming/keratorch/.venv/lib/python3.12/site-packages/keras/src/optimizers/base_optimizer.py:774: UserWarning: Gradients do not exist for variables ['variable_1', 'variable_2'] when minimizing the loss. If using `model.compile()`, did you forget to provide a `loss` argument?
  warnings.warn(
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - loss: 1.6689 - mae: 1.0293 - mse: 1.6598
Epoch 2/5
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - loss: 1.6614 - mae: 1.0263 - mse: 1.6524
Epoch 3/5
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 1.6542 - mae: 1.0233 - mse: 1.6452 
Epoch 4/5
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 1.6470 - mae: 1.0204 - mse: 1.6380
Epoch 5/5
4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 1.6398 - mae: 1.0175 - mse: 1.6308
```