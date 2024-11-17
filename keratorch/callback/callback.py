from typing import Dict, Any

from .state import State

__all__ = ["CallBack", "CallBackList"]


class CallBack:

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, state: State = None):
        pass 

    def on_batch_begin(self, state: State = None):
        pass 

    def on_batch_end(self, state: State = None):
        pass 

    def on_epoch_end(self, state: State = None):
        pass 


class CallBackList(CallBack):

    def __init__(self, state: State = None ):
        super(CallBackList, self).__init__()

        self.callbacks: list[CallBack] = []
        self.state: State = state

    def append(self, *callback: tuple[CallBack]):
        for call_back in callback: 
            self.callbacks.append(call_back)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(state=self.state)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(state=self.state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(state=self.state)

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end(state=self.state)

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(state=self.state)
