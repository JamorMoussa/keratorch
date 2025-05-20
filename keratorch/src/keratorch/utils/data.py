from torch.utils.data import (
    DataLoader as TorchDataLoader, Dataset
)
import torch 
from typing import Any


class DataLoader(TorchDataLoader):

    def __init__(
        self, 
        dataset: Dataset,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._keratorch_collate_fn,
            **kwargs
        )

    def _keratorch_collate_fn(self, batch: list[tuple[dict[str, Any], Any]]):

        # Separate inputs and targets
        inputs, targets = zip(*batch)
                              
        collated_inputs = {
            key: torch.stack([inp[key] for inp in inputs])
            for key in inputs[0]
        }

        # Collate targets
        targets = torch.stack(targets)

        return collated_inputs, targets

