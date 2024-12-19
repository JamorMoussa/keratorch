from .callback import CallBack
from ..metrics import Metric

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import ktState


__all__ = ["CallBackList", ]



class _callbacklist(CallBack):

    def __init__(self, state: "ktState"):

        self.list: list[CallBack] = []
        self.state = state 

    def append(self, *callback: tuple[CallBack], is_merics= False):
        for call_back in callback:
            if is_merics: 
                if not isinstance(call_back, Metric):
                    raise ValueError(f"'metrics' arguments in the 'ktTrainer.compile' must be list[Metric], got {type(call_back)} instead.")
            elif not isinstance(call_back, CallBack) or isinstance(call_back, Metric):
                raise ValueError(f"'callbacks' arguments in the 'ktTrainer.compile' must be 'list[CallBack]', got '{call_back.__class__.__name__}' instead.")
            self.list.append(call_back)

    def clear(self):
        self.list.clear()

    def on_train_begin(self):
        for callback in self.list:
            callback.on_train_begin(state=self.state)

    def on_epoch_begin(self):
        for callback in self.list:
            callback.on_epoch_begin(state=self.state)

    def on_batch_begin(self):
        self.update_recordsflags()
        for callback in self.list:
            callback.on_batch_begin(state=self.state)

    def on_batch_end(self):
        for callback in self.list: 
            callback.on_batch_end(state=self.state)

    def on_epoch_end(self):
        for callback in self.list:
            callback.on_epoch_end(state=self.state)


    def update_recordsflags(self):
        total = self.state.train.loadersize * self.state.hyparams.epochs
    
        p_train = int(total / self.state.hyparams.num_records)
        iter_ = self.state.hyparams.itr + self.state.hyparams.epoch * self.state.train.loadersize 

        if (iter_ % p_train == 0) and (iter_ != 0):
            self.state.hyparams.record_flag = True 
        else: 
            self.state.hyparams.record_flag = False

class CallBackList:

    def __init__(self, state: "ktState"):
        super(CallBackList, self).__init__()

        self.train = _callbacklist(state=state)
        # self.eval = _CallBackList(state=state)
        self.state = state
