from ..callbacks import CallBack

from abc import ABC 
from typing import Callable

__all__ = ["Metric", ]

class Metric(CallBack, ABC):

    def __init__(
        self, name: str, metric_func: Callable = None 
    ):
        super().__init__()

        self.name = name 
        self.metric_func = metric_func

        self.value = 0
        self.counter = 1

    def reset(self):
        self.value = 0
        self.counter = 1

    def on_batch_end(self, state):

        if self.metric_func is not None and state.hyparams.record_flag: 
            self.value += (self.metric_func(state= state) / self.counter)
            self.counter += 1

            state.history.update(
                name= self.name, value= self.value 
            )
            state.metrics.update_metric(
                name= self.name, value= self.value 
            )
        else:
            self.reset()


