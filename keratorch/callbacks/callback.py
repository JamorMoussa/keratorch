from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import ktState

__all__ = ["CallBack", ]


class CallBack:

    def on_train_begin(self, state: "ktState"):
        pass

    def on_epoch_begin(self, state: "ktState"):
        pass 

    def on_batch_begin(self, state: "ktState"):
        pass 

    def on_batch_end(self, state: "ktState"):
        pass 

    def on_epoch_end(self, state: "ktState"):
        pass 
