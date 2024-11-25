from typing import TYPE_CHECKING

from ..state import State

if TYPE_CHECKING:
    from .history import History

__all__ = ["CallBack", "CallBackList"]


class CallBack:

    def on_train_begin(self, state: State):
        pass

    def on_epoch_begin(self, state: State):
        pass 

    def on_batch_begin(self, state: State):
        pass 

    def on_batch_end(self, state: State):
        pass 

    def on_epoch_end(self, state: State):
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
        self.update_recordsflags()
        for callback in self.callbacks:
            callback.on_batch_begin(state=self.state)

    def on_batch_end(self):
        for callback in self.callbacks: 
            callback.on_batch_end(state=self.state)

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(state=self.state)

    def update_recordsflags(self):

        total = self.state.hyparams.loadersize * self.state.hyparams.num_iters
        
        p_train = int(total / self.state.hyparams.num_records)

        p_val = int(total / self.state.val.val_records)

        iter_ = self.state.hyparams.iter + self.state.hyparams.epoch * self.state.hyparams.loadersize 

        if (iter_ % p_train == 0) and (iter_ != 0):
            self.state.record_flag = True 
        else: 
            self.state.record_flag = False

        if (iter_ % p_val == 0) and (iter_ != 0):
            self.state.val.records_flag = True 
        else:
            self.state.val.records_flag = False 