from dataclasses import dataclass, field

import torch 

__all__ = ["ModelOutput", ]

@dataclass
class ModelOutput:

    outputs: torch.Tensor = field(default_factory= lambda: None)
    
    others: dict[str, torch.Tensor] = field(default_factory= lambda: {})
    