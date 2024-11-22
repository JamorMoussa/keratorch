from tqdm import tqdm 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TqdmIterator:

    def __init__(self):

        self.loader: "DataLoader"
        self.tqdm_iter: tqdm
        self.desc: str = None

        self.metrics: dict[str, str] = {}


    def get_tqdm(self, loader: "DataLoader", enum=False):
        self.loader = loader
        self.tqdm_iter = tqdm(iterable=loader)

        return enumerate(self.tqdm_iter) if enum else self.tqdm_iter

    def update(self):
        self.desc = " | ".join([f"{key}: {val}" for key, val in self.metrics.items()])
        self.tqdm_iter.set_description(self.desc)

