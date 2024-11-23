from typing import Dict, Any, List
from collections import defaultdict

from .callback import CallBack
from ..state import State

__all__ = ["History", ]

class History(CallBack):

    def __init__(self):
        self.history = defaultdict(list)

    def clear(self):
        self.history.clear()
    
    def on_train_begin(self, state: State):
        self.history.clear()

    def on_batch_end(self, state: State = None):
        state.tqdm_iter._metrics["Epoch"] = f"[{state.hyparams.epoch}/{state.hyparams.num_iters}]"
    