from typing import Dict, Any, List
from collections import defaultdict

from .callback import CallBack

__all__ = ["History", ]

class History(CallBack):

    def __init__(self):
        self.history = defaultdict(list)
    
    def on_train_begin(self, logs: Dict[str, Any] = None):
        self.history.clear()
    
    def on_epoch_end(self, epoch, logs: Dict[str, Any] = dict()):
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def on_batch_end(self, batch, logs: Dict[str, Any] = dict()):
        for key, value in logs.items():
            self.history[key].append(value)
