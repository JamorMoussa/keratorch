import keratorch as kt
import keratorch.nn as ktnn 

import torch, torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyDataset(Dataset):

    def __init__(self, n_rows: int = 100):
        super().__init__()

        self.n_rows = n_rows

        self.x = torch.randint(0, 30, (n_rows, 3))
        self.y = torch.randint(0, 3, (n_rows, ))

    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, index):
        return (
            {"user_ids": self.x[index]}, self.y[index]
        )
    

class UserEmbeddingModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.user_embed = nn.Embedding(num_embeddings=30, embedding_dim=12)

        self.fc = nn.Linear(in_features=12, out_features=3)

    def forward(self, user_ids):

        out = self.user_embed(user_ids).sum(dim=1) # (B, C)

        return self.fc(out)


dataset = MyDataset(n_rows=1000) 

dataloader = kt.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = ktnn.build_model_from(
    torch_module= UserEmbeddingModel()
)

model.compile(
    optimizer="adam", loss= ktnn.CrossEntropyLoss(), metrics =["accuracy"]
)

history = model.fit(dataloader, epochs=3)

