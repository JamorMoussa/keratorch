from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .train import ktTrainer

import torch

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
        self, trainloader: "DataLoader", num_iters: int, num_records: int = None, val_split: float = None
    ):
        self.train()
        
        self.update_state_params_before_training(
            num_iters=num_iters, loadersize=len(trainloader), num_records=num_records, batch_size=trainloader.batch_size
        )

        train_loader, val_loader = self.get_loaders(
            trainloader=trainloader, val_split=val_split
        )
        
        self.callbacklist.on_train_begin()

        for epoch in range(num_iters):
            
            self.callbacklist.on_epoch_begin()

            for itr, batch in self.tqdm_iter.get_tqdm(loader=train_loader, as_enumerate=True):
                
                self.update_state_params_after_iter(epoch=epoch, itr=itr)

                self.callbacklist.on_batch_begin()

                outputs, targets = self.do_forward_pass(batch=batch)
                
                loss = self.compute_loss(
                    outputs=outputs, targets=targets
                )
                
                self.do_backward_pass(loss=loss)

                if self.state.record_flag and self.state.val.do_validation:
                    self.do_validation(
                        val_loader = val_loader
                    )

                self.callbacklist.on_batch_end()

            self.callbacklist.on_epoch_end()

        return self.history

    def evaluate(
        self, testloader: "DataLoader"
    ):
        # TODO: This just a vanilla implimentation, shouldn't hard code the eval_loss #2
        # Thihs sould follow the fit design.   
        
        self.eval()

        eval_loss = 0

        with torch.no_grad():
            
            for itr, batch in self.tqdm_iter.get_tqdm(loader=testloader, as_enumerate=True):

                outputs, targets = self.do_forward_pass(batch=batch)

                loss = self.compute_loss(
                    outputs=outputs, targets=targets
                )

                eval_loss += loss.item() / len(batch[0])

        eval_loss /= len(testloader)
        
        return eval_loss


    def save(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


