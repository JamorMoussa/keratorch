from tqdm import tqdm 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from ..state import MetricState

__all__ = ["TqdmIterator", ]


class TqdmIterator:

    def __init__(self, metrics: 'MetricState'):
        super(TqdmIterator, self).__init__()

        self.metrics = metrics
        self.metrics.update(tqdm_iter= self)
        self.tqdm_iter: tqdm = None 
        self.ndigit: int = 5
    
    def from_loader(
        self, loader: 'DataLoader', as_enumerate: bool = False 
    ):
        self.loader = loader
        self.tqdm_iter = tqdm(iterable=loader)

        return enumerate(self.tqdm_iter) if as_enumerate else self.tqdm_iter

    def update_metrics(self):
        
        def _format_value(val):
            return f"{val:.5f}" if val > 1e-5 else f"{val:.2e}"
            
        self.desc = " - ".join([f"{key}: {_format_value(val)}" for key, val in self.metrics.items()])
        self.tqdm_iter.set_description(self.desc)
