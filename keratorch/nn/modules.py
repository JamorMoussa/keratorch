from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .train import ktTrainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

__all__ = ["ktModule", ]


class ktModule(ktTrainer, ABC):

    def __init__(self):
        super(ktModule, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    def fit(
        self, trainloader: "DataLoader", num_iters: int, num_records: int = None
    ):
        
        self.update_state_params_before_training(
            num_iters=num_iters, loadersize=len(trainloader), num_records=num_records
        )
        
        self.callbacklist.on_train_begin()

        for epoch in range(num_iters):
            
            self.callbacklist.on_epoch_begin()

            for itr, batch in self.tqdm_iter.get_tqdm(loader=trainloader, enum=True):
                
                self.update_state_params_after_iter(epoch=epoch, itr=itr)

                self.callbacklist.on_batch_begin()

                outputs, targets = self.compute_forward(batch=batch)
                
                loss = self.compute_loss(
                    outputs=outputs, targets=targets
                )
                
                self.do_backward_optimizer_step(loss=loss)

                self.callbacklist.on_batch_end()

            self.callbacklist.on_epoch_end()

        return self.history

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


