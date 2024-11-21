from ..callbacks import CallBack, State

from abc import ABC, abstractmethod

__all__ = ["Metric", ]

class Metric(CallBack, ABC):

    def __init__(self, name: str):
        super(Metric, self).__init__()  
        self.name: str = name 
        self.reset()
    
    @abstractmethod
    def compute_value(self, state) -> float:
        ...

    def save_record(self, state: State):

        self.metric_value /= state.hyprams.loadersize

        state.tqdm_iter.metrics[self.name] = f"{self.metric_value:.4f}"
        state.tqdm_iter.update()
        state.history.history[self.name].append(
            self.metric_value
        )

    def reset(self):
        self.metric_value = 0
        self.counter = 1

    def on_batch_end(self, state: State):

        self.compute_value(state=state)

        if state.record_flag:
            self.save_record(state=state)
            self.reset()

