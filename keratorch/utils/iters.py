from tqdm import tqdm 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TqdmIterator:

    loader: "DataLoader"
    tqdm_iter: tqdm

    metrics: dict[str, str] = {}

    def __init__(self):
        pass

    def get_tqdm(self, loader: "DataLoader", enum=False):
        self.loader = loader
        self.tqdm_iter = tqdm(iterable=loader, desc=self.get_desc()) 

        return enumerate(self.tqdm_iter) if enum else self.tqdm_iter

    def get_desc(self,):
        return " | ".join([f"{key}: {val}" for key, val in self.metrics.items()])

