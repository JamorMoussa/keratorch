from typing import TYPE_CHECKING

from ..state import State

if TYPE_CHECKING:
    from .history import History

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

    def __init__(self, state: State, history: "History"):
        super(CallBackList, self).__init__()

        self.callbacks: list[CallBack] = []
        self.state: State = state
        self.history = history

        self.callbacks.append(self.history)

    def append(self, *callback: tuple[CallBack]):
        for call_back in callback: 
            self.callbacks.append(call_back)

    def clear(self):
        self.callbacks.clear()
        self.history.clear()
        self.state.tqdm_iter.clear()
        self.callbacks.append(self.history)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(state=self.state)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(state=self.state)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(state=self.state)

    def on_batch_end(self):
        self.update_recordflag()
        for callback in self.callbacks: 
            callback.on_batch_end(state=self.state)

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(state=self.state)

    def update_recordflag(self):

        total = self.state.hyparams.loadersize * self.state.hyparams.num_iters
        
        p = int(total / self.state.hyparams.num_records)

        iter_ = self.state.hyparams.iter + self.state.hyparams.epoch * self.state.hyparams.loadersize 

        if (iter_ % p == 0) and (iter_ != 0):
            self.state.record_flag = True 
        else: 
            self.state.record_flag = False
