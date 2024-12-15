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

    def on_train_begin(self, state):
        print(f"\nEpoch: [{state.hyparams.epoch}/{state.hyparams.num_iters}]")

    def on_epoch_end(self, state: State = None):
        print(f"\nEpoch: [{state.hyparams.epoch}/{state.hyparams.num_iters}]")
        # state.tqdm_iter._metrics["Epoch"] = f"[{state.hyparams.epoch}/{state.hyparams.num_iters}]"
    