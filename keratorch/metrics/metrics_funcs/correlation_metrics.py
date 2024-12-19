from ..metric import Metric

import torch


__all__ = ["PersonMetric", "person"]


@torch.no_grad()
def _person(state: "ktState"):

    outs = state.train.model_output.outputs.flatten()
    targets = state.train.batch[1].flatten()

    outs_ = outs - outs.mean()
    targets_ = targets - targets.mean()
    return (outs_ * targets_).sum().item() / ((outs_**2).sum().sqrt() * (targets_**2).sum().sqrt()).item()



class PersonMetric(Metric):

    def __init__(self):
        super(PersonMetric, self).__init__(
            name= "person", metric_func= _person
        )


def person():
    return PersonMetric()
