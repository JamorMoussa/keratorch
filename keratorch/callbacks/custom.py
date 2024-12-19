from .callback import CallBack


class EpochLogger(CallBack):

    def on_epoch_begin(self, state):
        print(f"\nEpoch: [{state.hyparams.epoch}/{state.hyparams.epochs}]")
