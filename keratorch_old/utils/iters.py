from tqdm import tqdm 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TqdmIterator:

    def __init__(self, ndigit: int=4):

        self.loader: "DataLoader"
        self.tqdm_iter: tqdm
        self.desc: str = None
        self.ndigit: int = ndigit 

        self._metrics: dict[str, str] = {}

    def set_metrics(self, name: str, value: float):
        self._metrics[name] = f"{value:.{self.ndigit}f}"
        self.update()


    def get_tqdm(self, loader: "DataLoader", as_enumerate=False):
        self.loader = loader
        self.tqdm_iter = tqdm(iterable=loader)

        return enumerate(self.tqdm_iter) if as_enumerate else self.tqdm_iter

    def update(self):
        self.desc = " - ".join([f"{key}: {val}" for key, val in self._metrics.items()])
        self.tqdm_iter.set_description(self.desc)

    def clear(self):
        self._metrics.clear()

