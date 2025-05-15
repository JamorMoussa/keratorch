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

# print(model(
#     torch.LongTensor([1, 2])
# ))


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

