from typing import Dict, Any, List
from collections import defaultdict

from .callback import CallBack
from .state import State

__all__ = ["History", ]

class History(CallBack):

    def __init__(self):
        self.history = defaultdict(list)
    
    def on_train_begin(self, state: State):
        self.history.clear()

    def on_batch_end(self, state: State = None):
        for key, value in state.logs.items():
            self.history[key].append(value)
    
    # def on_epoch_end(self, state: State):
    #     pass 

    # def on_batch_end(self, state: State):
    #     pass 
