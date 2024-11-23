from ..callbacks import CallBack, State

from abc import ABC, abstractmethod

__all__ = ["Metric", ]

class Metric(CallBack, ABC):

    def __init__(self, name: str):
        super(Metric, self).__init__()  
        self.name: str = name 

        self.metric_value = 0
        self.counter = 1
    
    @abstractmethod
    def compute_value(self, state: State) -> float:
        ...

    def save_record(self, state: State, metric_value: float):

        state.tqdm_iter.set_metrics(
            name=self.name, value=metric_value
        )
        state.history.history[self.name].append(metric_value)

    def reset(self):
        self.metric_value = 0
        self.counter = 1

    def on_batch_end(self, state: State):

        metric_value = self.compute_value(state=state)
        self.counter += 1

        if state.record_flag:
            metric_value /=  self.counter
            self.save_record(state=state, metric_value=metric_value)
            self.reset()

