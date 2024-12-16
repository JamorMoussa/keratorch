from ..state import ktState

__all__ = ["CallBack", "_CallBackList", "CallBackList"]


class CallBack:

    def on_train_begin(self, state: ktState):
        pass

    def on_epoch_begin(self, state: ktState):
        pass 

    def on_batch_begin(self, state: ktState):
        pass 

    def on_batch_end(self, state: ktState):
        pass 

    def on_epoch_end(self, state: ktState):
        pass 


class _CallBackList(CallBack):

    def __init__(self, state: ktState):

        self.list: list[CallBack] = []
        self.state = state 

    def append(self, *callback: tuple[CallBack]):
        for call_back in callback: 
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
        for callback in self.list:
            callback.on_batch_begin(state=self.state)

    def on_batch_end(self):
        for callback in self.list: 
            callback.on_batch_end(state=self.state)

    def on_epoch_end(self):
        for callback in self.list:
            callback.on_epoch_end(state=self.state)


class CallBackList:

    def __init__(self, state: ktState):
        super(CallBackList, self).__init__()

        self.train = _CallBackList(state=state)
        # self.eval = _CallBackList(state=state)
        self.state = state
