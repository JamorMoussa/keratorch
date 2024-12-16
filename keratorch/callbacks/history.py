from .callback import CallBack

from collections import defaultdict

__all__ = ["History", ]

class History(CallBack):

    def __init__(self):
        self.history = defaultdict(list)

    def clear(self):
        self.history.clear()

    def update(self, name: str, value: 'Any'):
        self.history[name].append(value)
    
    def on_train_begin(self, state):
        self.clear()