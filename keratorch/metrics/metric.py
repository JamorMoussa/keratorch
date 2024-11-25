from ..callbacks import CallBack, State

from abc import ABC, abstractmethod

__all__ = ["Metric", ]

class Metric(CallBack, ABC):

    def __init__(self, name: str):
        super(Metric, self).__init__()  
        self.name: str = name 

        self.train_value = 0
        self.val_value = 0
        self.counter = 1
    
    @abstractmethod
    def compute_train_metric(self, state: State) -> None:
        ...

    @abstractmethod
    def compute_val_metric(self, state: State) -> None:
        ...

    def save_record(self, state: State, name: str, metric_value: float):

        state.tqdm_iter.set_metrics(
            name=name, value=metric_value
        )
        state.history.history[name].append(metric_value)

    def reset(self):
        self.train_value = 0
        self.val_value = 0
        self.counter = 1

    def on_batch_end(self, state: State):
    
        self.compute_train_metric(state=state)
        self.counter += 1

        if state.val.do_validation and state.val.records_flag:
            train_value = self.train_value / self.counter
            self.compute_val_metric(state=state)
            self.save_record(state=state, name="val_" + self.name, metric_value=self.val_value)
            self.save_record(state=state, name="train_" + self.name, metric_value=train_value)

        if state.record_flag:
            self.train_value /=  self.counter
            self.save_record(state=state, name=self.name, metric_value=self.train_value)
            self.reset()

