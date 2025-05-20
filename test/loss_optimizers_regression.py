import keratorch as kt

import torch, torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyDataset(Dataset):

    def __init__(self, n_rows: int = 100):
        super().__init__()

        self.n_rows = n_rows

        self.x = torch.rand(n_rows, 3)
        self.y = torch.rand(n_rows, 1)

    def __len__(self):
        return self.n_rows
    
    def __getitem__(self, index):
        return (
            {"input": self.x[index]}, self.y[index]
        )

dataset = MyDataset(n_rows=1000) 

dataloader = kt.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = kt.nn.build_model_from(
    torch_module= nn.Linear(3, 1)
)

model.compile(
    optimizer="adam", loss=kt.nn.MSELoss()
)


print(model.torch_module.module.parameters())


model.fit(dataloader, epochs=3)


