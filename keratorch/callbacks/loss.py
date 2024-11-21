from typing import Dict, Any, List
from collections import defaultdict

from .callback import CallBack
from .state import State

__all__ = ["LossCallBack", ]

class LossCallBack(CallBack):

    loss: float 
    counter: int = 1  

    def __init__(self):
        super(LossCallBack, self).__init__()

        self.reset()

    def reset(self):
        self.loss = 0
        self.counter = 1

    def update(self, loss: float):
        self.loss += loss
        self.counter += 1

    def get_loss(self):
        return self.loss / self.counter
    
    def on_batch_end(self, state: State = None):

        if state.record_flag:
            state.history.history["train_loss"].append(self.get_loss())
            self.reset()
        else:
            self.update(loss=state.loss)