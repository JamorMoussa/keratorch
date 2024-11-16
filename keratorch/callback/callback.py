from typing import Dict, Any

__all__ = ["CallBack", "CallBackList"]


class CallBack:

    def on_train_begin(self):
        pass

    def on_epoch_begin(self, epoch=None, logs: Dict[str, Any] = None):
        pass 

    def on_batch_begin(self, batch=None, logs: Dict[str, Any] = None):
        pass 

    def on_batch_end(self, batch=None, logs: Dict[str, Any] = None):
        pass 

    def on_epoch_end(self, epoch=None, logs: Dict[str, Any] = None):
        pass 


class CallBackList(CallBack):

    def __init__(self):
        super(CallBackList, self).__init__()

        self.callbacks: list[CallBack] = []

    def append(self, *callback: tuple[CallBack]):
        for call_back in callback: 
            self.callbacks.append(call_back)

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_epoch_begin(self, epoch=None, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_batch_begin(self, batch=None, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch=None, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_epoch_end(self, epoch=None, logs: Dict[str, Any] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
